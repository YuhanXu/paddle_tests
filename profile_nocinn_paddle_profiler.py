"""
单步 No-CINN 训练 + Paddle Profiler（用于对比 CINN 版本）
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
from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM

paddle.set_device('gpu:0')
paddle.set_flags({
    "FLAGS_print_ir": False,
    "FLAGS_deny_cinn_ops": "",
})


def create_small_llama(hidden_size=1024, num_layers=8, num_heads=8):
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 3,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        max_position_embeddings=512,
        use_cache=False,
    )
    model = LlamaForCausalLM(config)
    num_params = sum(p.numel().item() for p in model.parameters())
    print(f"--[Model] Small Llama: hidden={hidden_size}, layers={num_layers}, "
          f"heads={num_heads}, params={num_params/1e6:.1f}M")
    return model


if __name__ == "__main__":
    print(f"Paddle version: {paddle.__version__}")
    print(f"Device: {paddle.get_device()}")

    paddle.seed(42)
    model = create_small_llama()
    model.train()

    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    input_ids = paddle.randint(0, 32000, [1, 128])
    labels = paddle.randint(0, 32000, [1, 128])

    # Warmup
    print("\n--[Warmup] Running 3 warmup steps...")
    for i in range(3):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        print(f"  warmup step {i+1}: loss={loss.item():.6f}")
    paddle.device.synchronize()
    print("--[Warmup] Done.\n")

    # Profiled steps
    print("--[Profile] Starting profiled training steps (NO CINN)...")

    prof = paddle.profiler.Profiler(
        targets=[paddle.profiler.ProfilerTarget.CPU, paddle.profiler.ProfilerTarget.GPU],
        scheduler=paddle.profiler.make_scheduler(
            closed=0, ready=0, record=5, repeat=1
        ),
        on_trace_ready=paddle.profiler.export_chrome_tracing('./nocinn_profile_output'),
        timer_only=False,
    )

    prof.start()
    for step in range(5):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        paddle.device.synchronize()
        prof.step()
        print(f"  profiled step {step+1}: loss={loss.item():.6f}")
    prof.stop()

    print("\n--[Profile Summary - NO CINN]")
    prof.summary(
        op_detail=True,
        thread_sep=False,
        time_unit='ms',
    )
    print("\n--[Done]")
