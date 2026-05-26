"""
Ernie / Bert / GPT-2 单步 Profile：CINN vs No-CINN
使用 Paddle Profiler 收集 kernel 级别统计

Usage:
    python profile_nlp_models.py --model ernie
    python profile_nlp_models.py --model bert
    python profile_nlp_models.py --model gpt2
    python profile_nlp_models.py --model ernie --use_cinn    # 只跑 CINN 版
    python profile_nlp_models.py --model ernie --no_cinn     # 只跑 No-CINN 版
"""
import sys
sys.path = [p for p in sys.path if '/work/Paddle' not in p]
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
sys.path.insert(0, '/work/env3.10/lib/python3.10/site-packages')

import time
import os
os.environ["FLAGS_prim_all"] = "true"
import numpy as np
import paddle
from paddle import nn
import paddlenlp

paddle.set_device('gpu:0')
paddle.set_flags({
    "FLAGS_print_ir": False,
    "FLAGS_deny_cinn_ops": "",
})


def to_cinn_net(net, **kwargs):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = True
    return paddle.jit.to_static(
        net,
        build_strategy=build_strategy,
        full_graph=True,
        **kwargs
    )


def create_model(model_name):
    """Create model by name"""
    if model_name == "ernie":
        model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained(
            'ernie-3.0-nano-zh', num_classes=2)
        display = "Ernie-3.0-nano-zh"
    elif model_name == "bert":
        model = paddlenlp.transformers.BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_classes=2)
        display = "Bert-base-uncased"
    elif model_name == "gpt2":
        model = paddlenlp.transformers.GPTForSequenceClassification.from_pretrained(
            'gpt2-medium-en', num_classes=2)
        display = "GPT2-medium-en"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    num_params = sum(p.numel().item() for p in model.parameters())
    print(f"--[Model] {display}, params={num_params/1e6:.1f}M")
    return model


def run_profile(model_name, use_cinn, batch_size=1, seq_len=128, warmup_steps=3, profile_steps=5):
    """Run profiled training steps"""
    mode_str = "cinn" if use_cinn else "nocinn"
    print(f"\n{'='*60}")
    print(f"  Profiling {model_name} [{mode_str}] - {profile_steps} steps")
    print(f"{'='*60}")

    paddle.seed(42)
    model = create_model(model_name)

    if use_cinn:
        net = to_cinn_net(model)
    else:
        net = model

    net.train()
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    input_ids = paddle.randint(0, 1000, [batch_size, seq_len])
    labels = paddle.randint(0, 2, [batch_size])

    def forward_fn():
        outputs = net(input_ids=input_ids, labels=labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        return loss

    # Warmup
    print(f"\n--[Warmup] Running {warmup_steps} warmup steps...")
    for i in range(warmup_steps):
        loss = forward_fn()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        print(f"  warmup step {i+1}: loss={loss.item():.6f}")
    paddle.device.synchronize()
    print("--[Warmup] Done.\n")

    # Profile
    print(f"--[Profile] Starting {profile_steps} profiled steps...")
    output_dir = f'./{model_name}_{mode_str}_profile_output'

    prof = paddle.profiler.Profiler(
        targets=[paddle.profiler.ProfilerTarget.CPU, paddle.profiler.ProfilerTarget.GPU],
        scheduler=paddle.profiler.make_scheduler(
            closed=0, ready=0, record=profile_steps, repeat=1
        ),
        on_trace_ready=paddle.profiler.export_chrome_tracing(output_dir),
        timer_only=False,
    )

    prof.start()
    for step in range(profile_steps):
        loss = forward_fn()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        paddle.device.synchronize()
        prof.step()
        print(f"  profiled step {step+1}: loss={loss.item():.6f}")
    prof.stop()

    print(f"\n--[Profile Summary - {model_name} {mode_str}]")
    prof.summary(
        op_detail=True,
        thread_sep=False,
        time_unit='ms',
    )

    print(f"\n--[Done] Trace saved to: {output_dir}/")

    # Cleanup
    del model, net, optimizer
    paddle.device.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLP Models Profile: CINN vs No-CINN")
    parser.add_argument("--model", type=str, required=True,
                        choices=["ernie", "bert", "gpt2"],
                        help="Which model to profile")
    parser.add_argument("--use_cinn", action="store_true", help="Only profile CINN version")
    parser.add_argument("--no_cinn", action="store_true", help="Only profile No-CINN version")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile_steps", type=int, default=5)
    args = parser.parse_args()

    print(f"Paddle version: {paddle.__version__}")
    print(f"Device: {paddle.get_device()}")

    # Default: run both
    run_nocinn = not args.use_cinn
    run_cinn = not args.no_cinn

    if run_nocinn:
        run_profile(args.model, use_cinn=False,
                   batch_size=args.batch_size, seq_len=args.seq_len,
                   warmup_steps=args.warmup, profile_steps=args.profile_steps)

    if run_cinn:
        run_profile(args.model, use_cinn=True,
                   batch_size=args.batch_size, seq_len=args.seq_len,
                   warmup_steps=args.warmup, profile_steps=args.profile_steps)
