"""
Llama2 训练 Benchmark：CINN vs No-CINN 对比
使用小型 Llama 配置 (168M) 以确保 CINN 训练不 OOM
跑 500 步，收集 loss 并画对比图
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
    """创建一个小型 Llama 模型用于 CINN 训练对比"""
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 3,  # ~2.75x, 类似 Llama 比例
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


def train_n_steps(net, input_ids, labels, optimizer, total_steps=500,
                  log_interval=10, mode_str="nocinn"):
    """训练 N 步并收集 loss 和时间信息"""
    net.train()
    losses = []
    step_times = []
    total_start = time.time()

    for step in range(total_steps):
        t1 = time.time()

        outputs = net(input_ids=input_ids, labels=labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        paddle.device.synchronize()

        t2 = time.time()
        step_time_ms = (t2 - t1) * 1000
        loss_val = loss.item()

        losses.append(loss_val)
        step_times.append(step_time_ms)

        if (step + 1) % log_interval == 0 or step == 0:
            avg_time = np.mean(step_times[-log_interval:])
            print(f"  [{mode_str}] step [{step+1:>4d}/{total_steps}] "
                  f"loss: {loss_val:.6f} | "
                  f"avg_time: {avg_time:.2f} ms/step | "
                  f"throughput: {1000.0/avg_time:.2f} steps/s")

    total_time = time.time() - total_start

    # 统计汇总
    print(f"\n  --[{mode_str} Summary] {total_steps} steps in {total_time:.1f}s")
    print(f"  --[Loss] initial: {losses[0]:.6f} | final: {losses[-1]:.6f}")
    print(f"  --[Time] avg: {np.mean(step_times):.2f} ms | "
          f"p50: {np.percentile(step_times, 50):.2f} ms | "
          f"p95: {np.percentile(step_times, 95):.2f} ms")
    print(f"  --[Throughput] avg: {1000.0/np.mean(step_times):.2f} steps/s")

    return {"losses": losses, "step_times": step_times, "total_time": total_time}


def save_csv(results, filename):
    """保存结果到 CSV"""
    with open(filename, 'w') as f:
        f.write("step,loss,step_time_ms\n")
        for i, (loss, t) in enumerate(zip(results["losses"], results["step_times"])):
            f.write(f"{i+1},{loss:.6f},{t:.2f}\n")
    print(f"  --[Save] {filename}")


def plot_comparison(nocinn_results, cinn_results, output_path="llama2_train_loss_comparison.png"):
    """画 loss 对比曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = range(1, len(nocinn_results["losses"]) + 1)

    # Loss 曲线对比
    ax1.plot(steps, nocinn_results["losses"], label="No CINN (eager)", alpha=0.8, linewidth=1.2)
    ax1.plot(steps, cinn_results["losses"], label="CINN (compiled)", alpha=0.8, linewidth=1.2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss: CINN vs No-CINN")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Step time 对比
    ax2.plot(steps, nocinn_results["step_times"], label="No CINN (eager)", alpha=0.6, linewidth=0.8)
    ax2.plot(steps, cinn_results["step_times"], label="CINN (compiled)", alpha=0.6, linewidth=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Step Time (ms)")
    ax2.set_title("Step Time: CINN vs No-CINN")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 在 step time 图上标注平均值
    nocinn_avg = np.mean(nocinn_results["step_times"][10:])  # 跳过前几步warmup
    cinn_avg = np.mean(cinn_results["step_times"][10:])
    ax2.axhline(y=nocinn_avg, color='C0', linestyle='--', alpha=0.5, label=f"No CINN avg: {nocinn_avg:.1f}ms")
    ax2.axhline(y=cinn_avg, color='C1', linestyle='--', alpha=0.5, label=f"CINN avg: {cinn_avg:.1f}ms")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n--[Plot] Saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Llama2 Training: CINN vs No-CINN Comparison")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Paddle version: {paddle.__version__}")
    print(f"Device: {paddle.get_device()}")
    print(f"Config: batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"steps={args.steps}, lr={args.lr}")
    print()

    # 固定随机种子，确保两个版本用相同初始权重和数据
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    # 生成输入数据
    input_ids = paddle.randint(0, 32000, [args.batch_size, args.seq_len])
    labels = paddle.randint(0, 32000, [args.batch_size, args.seq_len])

    # ============ No-CINN 训练 ============
    print("=" * 60)
    print("Phase 1: Training WITHOUT CINN (eager mode)")
    print("=" * 60)

    paddle.seed(args.seed)
    model_nocinn = create_small_llama(args.hidden_size, args.num_layers, args.num_heads)
    optimizer_nocinn = paddle.optimizer.AdamW(
        parameters=model_nocinn.parameters(),
        learning_rate=args.lr,
        weight_decay=0.01,
    )

    nocinn_results = train_n_steps(
        model_nocinn, input_ids, labels, optimizer_nocinn,
        total_steps=args.steps, log_interval=args.log_interval, mode_str="nocinn"
    )
    save_csv(nocinn_results, f"llama2_train_nocinn_small_{args.steps}steps.csv")

    # 释放显存
    del model_nocinn, optimizer_nocinn
    paddle.device.cuda.empty_cache()

    # ============ CINN 训练 ============
    print("\n" + "=" * 60)
    print("Phase 2: Training WITH CINN (compiled mode)")
    print("=" * 60)

    paddle.seed(args.seed)
    model_cinn = create_small_llama(args.hidden_size, args.num_layers, args.num_heads)
    model_cinn_compiled = to_cinn_net(model_cinn)
    optimizer_cinn = paddle.optimizer.AdamW(
        parameters=model_cinn.parameters(),
        learning_rate=args.lr,
        weight_decay=0.01,
    )

    cinn_results = train_n_steps(
        model_cinn_compiled, input_ids, labels, optimizer_cinn,
        total_steps=args.steps, log_interval=args.log_interval, mode_str="cinn"
    )
    save_csv(cinn_results, f"llama2_train_cinn_small_{args.steps}steps.csv")

    # ============ 画图对比 ============
    print("\n" + "=" * 60)
    print("Plotting comparison...")
    print("=" * 60)
    plot_comparison(nocinn_results, cinn_results)

    # 最终汇总
    nocinn_avg = np.mean(nocinn_results["step_times"][10:])
    cinn_avg = np.mean(cinn_results["step_times"][10:])
    speedup = nocinn_avg / cinn_avg
    print(f"\n{'=' * 60}")
    print(f"FINAL COMPARISON (skip first 10 steps)")
    print(f"  No-CINN avg step time: {nocinn_avg:.2f} ms")
    print(f"  CINN    avg step time: {cinn_avg:.2f} ms")
    print(f"  Speedup (CINN/NoCINN): {speedup:.2f}x")
    print(f"{'=' * 60}")
