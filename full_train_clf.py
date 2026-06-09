"""Full-train clf finetune for Bert / Ernie / GPT-2.

Public API: run_clf(model_key, task, hp, cinn_on, args) -> summary dict
"""
import os
import sys
import time

import paddle
from paddlenlp.transformers import (
    BertForSequenceClassification, ErnieForSequenceClassification,
    GPTForSequenceClassification, AutoTokenizer,
)

from full_train_data import build_loaders
from full_train_utils import (
    set_seed, maybe_wrap_cinn, make_lr_and_optimizer,
    open_csv_writer, append_csv, eval_clf,
    write_summary_json, StepTimer,
)


MODEL_REGISTRY = {
    'bert':  ('bert-base-uncased',   BertForSequenceClassification),
    'ernie': ('ernie-3.0-nano-zh',   ErnieForSequenceClassification),
    'gpt2':  ('gpt2-medium-en',      GPTForSequenceClassification),
}


def _create_model(model_key, num_labels):
    model_name, cls = MODEL_REGISTRY[model_key]
    model = cls.from_pretrained(model_name, num_classes=num_labels)
    return model_name, model


def _forward(model, batch, model_key):
    # filter None / drop token_type_ids when not supported
    feed = {}
    for k in ('input_ids', 'token_type_ids', 'attention_mask', 'position_ids'):
        if k in batch:
            feed[k] = batch[k]
    feed['labels'] = batch['labels']
    if model_key == 'gpt2':
        # GPT 没有 token_type_ids
        feed.pop('token_type_ids', None)
    out = model(**feed)
    loss = out[0] if isinstance(out, tuple) else out.loss
    logits = out[1] if isinstance(out, tuple) else out.logits
    return loss, logits


def run_clf(model_key, task, hp, cinn_on, args):
    """hp: dict(bs, seq, epochs, lr); args: argparse Namespace."""
    set_seed(args.seed)
    bs = args.batch_size or hp['bs']
    seq_len = args.seq_len or hp['seq']
    epochs = args.epochs or hp['epochs']
    lr = args.lr if args.lr is not None else hp['lr']

    model_name, model = _create_model(model_key, _ignored_num_labels := 2)
    # rebuild with correct num_labels after we read it from data
    data = build_loaders(task, model_name, bs, seq_len)
    num_labels = data['num_labels']
    model_name, model = _create_model(model_key, num_labels)

    train_loader, dev_loader = data['train'], data['dev']
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)

    sched, optim = make_lr_and_optimizer(model, total_steps, lr, args.warmup_ratio)
    fwd = maybe_wrap_cinn(model, cinn_on)

    cinn_tag = 'cinn' if cinn_on else 'nocinn'
    csv_path = os.path.join(args.output_dir, f'{model_key}_{task}_{cinn_tag}_steps.csv')
    f, w = open_csv_writer(csv_path, ['step', 'train_loss', 'train_acc',
                                       'dev_loss', 'dev_acc', 'step_time_ms', 'lr'])

    print(f'[run_clf] model={model_key} task={task} cinn={cinn_on} '
          f'bs={bs} seq={seq_len} epochs={epochs} lr={lr} total_steps={total_steps}')

    timer = StepTimer()
    global_step = 0
    best_acc = 0.0
    last_step_time = 0.0
    fwd.train()

    for epoch in range(epochs):
        for batch in train_loader:
            if global_step >= total_steps:
                break
            timer.tick()
            loss, logits = _forward(fwd, batch, model_key)
            loss.backward()
            optim.step()
            sched.step()
            optim.clear_grad()
            last_step_time = timer.tock()

            with paddle.no_grad():
                preds = paddle.argmax(logits, axis=-1)
                acc = float((preds == batch['labels']).astype('float32').mean().item())

            if global_step % args.log_interval == 0:
                print(f'  step {global_step}/{total_steps} loss={float(loss.item()):.4f} '
                      f'acc={acc:.4f} step_ms={last_step_time:.2f} lr={sched.get_lr():.2e}')

            dev_loss, dev_acc = '', ''
            do_eval = (args.eval_steps > 0 and global_step > 0
                       and global_step % args.eval_steps == 0)
            if do_eval:
                m = eval_clf(model, dev_loader, max_batches=args.eval_max_batches)
                dev_loss, dev_acc = m['dev_loss'], m['dev_acc']
                best_acc = max(best_acc, dev_acc)
                print(f'  [eval@{global_step}] dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f}')

            append_csv(w, [global_step, float(loss.item()), acc,
                           dev_loss, dev_acc, last_step_time, sched.get_lr()])
            global_step += 1
        if global_step >= total_steps:
            break
        # epoch-end eval
        m = eval_clf(model, dev_loader, max_batches=args.eval_max_batches)
        best_acc = max(best_acc, m['dev_acc'])
        print(f'  [epoch {epoch} end] dev_loss={m["dev_loss"]:.4f} dev_acc={m["dev_acc"]:.4f}')
        append_csv(w, [global_step, '', '', m['dev_loss'], m['dev_acc'], '', sched.get_lr()])

    # final eval
    m = eval_clf(model, dev_loader, max_batches=args.eval_max_batches)
    best_acc = max(best_acc, m['dev_acc'])
    f.close()

    summary = dict(
        model=model_key, task=task, cinn=cinn_on,
        total_steps=global_step, dev_acc=m['dev_acc'], best_dev_acc=best_acc,
        last_step_time_ms=last_step_time,
        hp=dict(bs=bs, seq=seq_len, epochs=epochs, lr=lr),
    )
    write_summary_json(
        os.path.join(args.output_dir, f'{model_key}_{task}_{cinn_tag}_summary.json'),
        summary,
    )
    return summary
