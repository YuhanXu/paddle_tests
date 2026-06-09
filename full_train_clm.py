"""Full-train CLM/SFT for Small Llama 168M.

Public API: run_clm(task, hp, cinn_on, args) -> summary dict.
Builds the same architecture as benchmark_train_llama2_compare.py:
    8L / 1024H / 8 heads / vocab=32000 / max_pos=512.
"""
import os

import paddle
from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM

from full_train_data import build_loaders
from full_train_utils import (
    set_seed, maybe_wrap_cinn, make_lr_and_optimizer,
    open_csv_writer, append_csv, eval_clm,
    write_summary_json, StepTimer,
)


# Need a tokenizer name that loads (LlamaTokenizer) — we still create the
# config-driven model but reuse a tokenizer for tokenization on real text.
LLAMA_TOKENIZER_NAME = os.environ.get('FULL_TRAIN_LLAMA_TOKENIZER',
                                       'facebook/llama-7b')


def create_small_llama(hidden_size=1024, num_layers=8, num_heads=8,
                       vocab_size=32000, max_pos=512):
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 3,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        max_position_embeddings=max_pos,
        use_cache=False,
    )
    return LlamaForCausalLM(config)


def run_clm(task, hp, cinn_on, args):
    set_seed(args.seed)
    bs = args.batch_size or hp['bs']
    seq_len = args.seq_len or hp['seq']
    epochs = args.epochs or hp['epochs']
    lr = args.lr if args.lr is not None else hp['lr']

    model = create_small_llama(max_pos=max(512, seq_len))
    data = build_loaders(task, LLAMA_TOKENIZER_NAME, bs, seq_len)
    train_loader, dev_loader = data['train'], data['dev']

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)

    warmup_ratio = args.warmup_ratio
    if task == 'wikitext103':
        # 论文按 step 数 warmup（2000），换算成比例
        warmup_ratio = min(0.1, 2000 / max(1, total_steps))

    sched, optim = make_lr_and_optimizer(model, total_steps, lr, warmup_ratio)
    fwd = maybe_wrap_cinn(model, cinn_on)

    cinn_tag = 'cinn' if cinn_on else 'nocinn'
    csv_path = os.path.join(args.output_dir, f'llama_{task}_{cinn_tag}_steps.csv')
    f, w = open_csv_writer(csv_path,
                           ['step', 'train_loss', 'dev_loss', 'dev_ppl',
                            'step_time_ms', 'lr'])

    print(f'[run_clm] task={task} cinn={cinn_on} bs={bs} seq={seq_len} '
          f'epochs={epochs} lr={lr} total_steps={total_steps}')

    timer = StepTimer()
    global_step = 0
    best_ppl = float('inf')
    last_step_time = 0.0
    fwd.train()

    for epoch in range(epochs):
        for batch in train_loader:
            if global_step >= total_steps:
                break
            timer.tick()
            out = fwd(input_ids=batch['input_ids'], labels=batch['labels'])
            loss = out[0] if isinstance(out, tuple) else out.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.clear_grad()
            last_step_time = timer.tock()

            if global_step % args.log_interval == 0:
                print(f'  step {global_step}/{total_steps} loss={float(loss.item()):.4f} '
                      f'step_ms={last_step_time:.2f} lr={sched.get_lr():.2e}')

            dev_loss, dev_ppl = '', ''
            do_eval = (args.eval_steps > 0 and global_step > 0
                       and global_step % args.eval_steps == 0)
            if do_eval:
                m = eval_clm(model, dev_loader, max_batches=args.eval_max_batches)
                dev_loss, dev_ppl = m['dev_loss'], m['dev_ppl']
                best_ppl = min(best_ppl, dev_ppl)
                print(f'  [eval@{global_step}] dev_loss={dev_loss:.4f} dev_ppl={dev_ppl:.2f}')

            append_csv(w, [global_step, float(loss.item()),
                           dev_loss, dev_ppl, last_step_time, sched.get_lr()])
            global_step += 1
        if global_step >= total_steps:
            break
        m = eval_clm(model, dev_loader, max_batches=args.eval_max_batches)
        best_ppl = min(best_ppl, m['dev_ppl'])
        print(f'  [epoch {epoch} end] dev_loss={m["dev_loss"]:.4f} dev_ppl={m["dev_ppl"]:.2f}')
        append_csv(w, [global_step, '', m['dev_loss'], m['dev_ppl'], '', sched.get_lr()])

    m = eval_clm(model, dev_loader, max_batches=args.eval_max_batches)
    best_ppl = min(best_ppl, m['dev_ppl'])
    f.close()

    summary = dict(
        model='llama', task=task, cinn=cinn_on,
        total_steps=global_step,
        dev_loss=m['dev_loss'], dev_ppl=m['dev_ppl'], best_dev_ppl=best_ppl,
        last_step_time_ms=last_step_time,
        hp=dict(bs=bs, seq=seq_len, epochs=epochs, lr=lr),
    )
    write_summary_json(
        os.path.join(args.output_dir, f'llama_{task}_{cinn_tag}_summary.json'),
        summary,
    )
    return summary
