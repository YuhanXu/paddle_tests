"""
单步 CINN 训练 + Paddle Profiler
使用小型 Llama (168M) 模型，warmup 后 profile 1 step 训练
输出 Chrome tracing JSON 和 kernel 统计
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


def to_cinn_net(net, **kwargs):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = True
    return paddle.jit.to_static(
        net,
        build_strategy=build_strategy,
        full_graph=True,
        **kwargs
    )


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

    # 创建模型和数据
    paddle.seed(42)
    model = create_small_llama()
    model_cinn = to_cinn_net(model)
    model_cinn.train()

    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    input_ids = paddle.randint(0, 32000, [1, 128])
    labels = paddle.randint(0, 32000, [1, 128])

    # Warmup: 跑几步让 CINN 完成编译
    print("\n--[Warmup] Running 3 warmup steps (CINN compilation happens here)...")
    for i in range(3):
        outputs = model_cinn(input_ids=input_ids, labels=labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        print(f"  warmup step {i+1}: loss={loss.item():.6f}")
    paddle.device.synchronize()
    print("--[Warmup] Done.\n")

    # Profiled steps: 用 Paddle Profiler 抓取
    print("--[Profile] Starting profiled training steps...")

    # 使用 Paddle Profiler
    prof = paddle.profiler.Profiler(
        targets=[paddle.profiler.ProfilerTarget.CPU, paddle.profiler.ProfilerTarget.GPU],
        scheduler=paddle.profiler.make_scheduler(
            closed=0,   # 不跳过
            ready=0,    # 不预热（已经warmup过了）
            record=5,   # 记录5步
            repeat=1
        ),
        on_trace_ready=paddle.profiler.export_chrome_tracing('./cinn_profile_output'),
        timer_only=False,
    )

    prof.start()
    for step in range(5):
        outputs = model_cinn(input_ids=input_ids, labels=labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        paddle.device.synchronize()
        prof.step()
        print(f"  profiled step {step+1}: loss={loss.item():.6f}")
    prof.stop()

    # 打印 profiler 汇总
    print("\n--[Profile Summary]")
    prof.summary(
        op_detail=True,
        thread_sep=False,
        time_unit='ms',
    )

    print("\n--[Done] Profile trace saved to: ./cinn_profile_output/")
    print("  You can view it in Chrome at chrome://tracing or https://ui.perfetto.dev")
