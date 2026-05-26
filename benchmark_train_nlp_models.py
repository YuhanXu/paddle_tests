"""
Ernie / Bert / GPT-2 训练 Benchmark：CINN vs No-CINN 对比
每个模型跑 500 步，收集 loss 信息并画对比图

Usage:
    python benchmark_train_nlp_models.py --model ernie --steps 500
    python benchmark_train_nlp_models.py --model bert --steps 500
    python benchmark_train_nlp_models.py --model gpt2 --steps 500
    python benchmark_train_nlp_models.py --model all --steps 500
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


def train_n_steps(net, forward_fn, optimizer, total_steps=500,
                  log_interval=10, mode_str="nocinn"):
    """
    训练 N 步并收集 loss 和时间信息
    forward_fn: callable that takes net and returns loss tensor
    """
    net.train()
    losses = []
    step_times = []
    total_start = time.time()

    for step in range(total_steps):
        t1 = time.time()

        loss = forward_fn(net)
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
    print(f"\n  --[{mode_str} Summary] {total_steps} steps in {total_time:.1f}s")
    print(f"  --[Loss] initial: {losses[0]:.6f} | final: {losses[-1]:.6f}")
    print(f"  --[Time] avg: {np.mean(step_times):.2f} ms | "
          f"p50: {np.percentile(step_times, 50):.2f} ms | "
          f"p95: {np.percentile(step_times, 95):.2f} ms")
    print(f"  --[Throughput] avg: {1000.0/np.mean(step_times):.2f} steps/s")

    return {"losses": losses, "step_times": step_times, "total_time": total_time}


def save_csv(results, filename):
    with open(filename, 'w') as f:
        f.write("step,loss,step_time_ms\n")
        for i, (loss, t) in enumerate(zip(results["losses"], results["step_times"])):
            f.write(f"{i+1},{loss:.6f},{t:.2f}\n")
    print(f"  --[Save] {filename}")


def plot_comparison(nocinn_results, cinn_results, model_name, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    steps = range(1, len(nocinn_results["losses"]) + 1)

    # Loss 曲线
    ax1.plot(steps, nocinn_results["losses"], label="No CINN (eager)", alpha=0.8, linewidth=1.2)
    ax1.plot(steps, cinn_results["losses"], label="CINN (compiled)", alpha=0.8, linewidth=1.2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} Training Loss: CINN vs No-CINN")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Step time 曲线
    ax2.plot(steps, nocinn_results["step_times"], label="No CINN (eager)", alpha=0.6, linewidth=0.8)
    ax2.plot(steps, cinn_results["step_times"], label="CINN (compiled)", alpha=0.6, linewidth=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Step Time (ms)")
    ax2.set_title(f"{model_name} Step Time: CINN vs No-CINN")

    nocinn_avg = np.mean(nocinn_results["step_times"][10:])
    cinn_avg = np.mean(cinn_results["step_times"][10:])
    ax2.axhline(y=nocinn_avg, color='C0', linestyle='--', alpha=0.5)
    ax2.axhline(y=cinn_avg, color='C1', linestyle='--', alpha=0.5)
    ax2.legend([f"No CINN (eager)", f"CINN (compiled)",
                f"No CINN avg: {nocinn_avg:.1f}ms", f"CINN avg: {cinn_avg:.1f}ms"])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  --[Plot] Saved to: {output_path}")
    plt.close()


# ================== Model Definitions ==================

class ModelBenchmark:
    """Base class for model training benchmarks"""

    def __init__(self, model_name, batch_size=1, seq_len=128, lr=1e-4, num_classes=2):
        self.model_name = model_name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lr = lr
        self.num_classes = num_classes

    def create_model(self):
        raise NotImplementedError

    def create_inputs(self):
        raise NotImplementedError

    def forward_fn(self, net):
        """Returns loss from one forward pass"""
        raise NotImplementedError

    def run_comparison(self, total_steps=500, log_interval=50):
        """Run CINN vs No-CINN comparison"""
        print(f"\n{'='*70}")
        print(f"  {self.model_name} Training Benchmark")
        print(f"  batch_size={self.batch_size}, seq_len={self.seq_len}, "
              f"steps={total_steps}, lr={self.lr}")
        print(f"{'='*70}")

        # Create inputs (shared)
        paddle.seed(42)
        np.random.seed(42)
        self.create_inputs()

        # ---- No-CINN ----
        print(f"\n  Phase 1: Training WITHOUT CINN (eager mode)")
        print(f"  {'-'*50}")
        paddle.seed(42)
        model_nocinn = self.create_model()
        model_nocinn.train()
        optimizer_nocinn = paddle.optimizer.AdamW(
            parameters=model_nocinn.parameters(),
            learning_rate=self.lr,
            weight_decay=0.01,
        )

        nocinn_results = train_n_steps(
            model_nocinn, self.forward_fn, optimizer_nocinn,
            total_steps=total_steps, log_interval=log_interval, mode_str="nocinn"
        )
        prefix = self.model_name.lower().replace('-', '_').replace(' ', '_')
        save_csv(nocinn_results, f"{prefix}_train_nocinn_{total_steps}steps.csv")

        del model_nocinn, optimizer_nocinn
        import gc; gc.collect()
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

        # ---- CINN ----
        print(f"\n  Phase 2: Training WITH CINN (compiled mode)")
        print(f"  {'-'*50}")
        paddle.seed(42)
        model_cinn_raw = self.create_model()
        model_cinn = to_cinn_net(model_cinn_raw)
        model_cinn_raw.train()
        optimizer_cinn = paddle.optimizer.AdamW(
            parameters=model_cinn_raw.parameters(),
            learning_rate=self.lr,
            weight_decay=0.01,
        )

        cinn_results = train_n_steps(
            model_cinn, self.forward_fn, optimizer_cinn,
            total_steps=total_steps, log_interval=log_interval, mode_str="cinn"
        )
        save_csv(cinn_results, f"{prefix}_train_cinn_{total_steps}steps.csv")

        # ---- Plot ----
        plot_comparison(nocinn_results, cinn_results, self.model_name,
                       f"{prefix}_train_loss_comparison.png")

        # ---- Summary ----
        nocinn_avg = np.mean(nocinn_results["step_times"][10:])
        cinn_avg = np.mean(cinn_results["step_times"][10:])
        speedup = nocinn_avg / cinn_avg if cinn_avg > 0 else 0
        print(f"\n  {'='*50}")
        print(f"  FINAL: {self.model_name}")
        print(f"    No-CINN avg step time: {nocinn_avg:.2f} ms")
        print(f"    CINN    avg step time: {cinn_avg:.2f} ms")
        print(f"    Speedup: {speedup:.2f}x")
        print(f"  {'='*50}")

        del model_cinn_raw, model_cinn, optimizer_cinn
        import gc; gc.collect()
        try:
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

        return {"nocinn": nocinn_results, "cinn": cinn_results, "speedup": speedup}


class ErnieBenchmark(ModelBenchmark):
    def __init__(self, batch_size=1, seq_len=128, lr=1e-4):
        super().__init__("Ernie-3.0-nano", batch_size, seq_len, lr, num_classes=2)

    def create_model(self):
        model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained(
            'ernie-3.0-nano-zh', num_classes=self.num_classes)
        num_params = sum(p.numel().item() for p in model.parameters())
        print(f"  --[Model] Ernie-3.0-nano-zh (SequenceClassification), params={num_params/1e6:.1f}M")
        return model

    def create_inputs(self):
        self.input_ids = paddle.randint(0, 1000, [self.batch_size, self.seq_len])
        self.labels = paddle.randint(0, self.num_classes, [self.batch_size])

    def forward_fn(self, net):
        outputs = net(input_ids=self.input_ids, labels=self.labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        return loss


class BertBenchmark(ModelBenchmark):
    def __init__(self, batch_size=1, seq_len=128, lr=1e-4):
        super().__init__("Bert-base-uncased", batch_size, seq_len, lr, num_classes=2)

    def create_model(self):
        model = paddlenlp.transformers.BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_classes=self.num_classes)
        num_params = sum(p.numel().item() for p in model.parameters())
        print(f"  --[Model] Bert-base-uncased (SequenceClassification), params={num_params/1e6:.1f}M")
        return model

    def create_inputs(self):
        self.input_ids = paddle.randint(0, 1000, [self.batch_size, self.seq_len])
        self.labels = paddle.randint(0, self.num_classes, [self.batch_size])

    def forward_fn(self, net):
        outputs = net(input_ids=self.input_ids, labels=self.labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        return loss


class GPT2Benchmark(ModelBenchmark):
    def __init__(self, batch_size=1, seq_len=128, lr=1e-4):
        super().__init__("GPT2-medium", batch_size, seq_len, lr, num_classes=2)

    def create_model(self):
        model = paddlenlp.transformers.GPTForSequenceClassification.from_pretrained(
            'gpt2-medium-en', num_classes=self.num_classes)
        num_params = sum(p.numel().item() for p in model.parameters())
        print(f"  --[Model] GPT2-medium-en (SequenceClassification), params={num_params/1e6:.1f}M")
        return model

    def create_inputs(self):
        self.input_ids = paddle.randint(0, 1000, [self.batch_size, self.seq_len])
        self.labels = paddle.randint(0, self.num_classes, [self.batch_size])

    def forward_fn(self, net):
        outputs = net(input_ids=self.input_ids, labels=self.labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        return loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLP Models Training Benchmark: CINN vs No-CINN")
    parser.add_argument("--model", type=str, default="all",
                        choices=["ernie", "bert", "gpt2", "all"],
                        help="Which model to benchmark")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    print(f"Paddle version: {paddle.__version__}")
    print(f"Device: {paddle.get_device()}")

    results = {}

    if args.model in ("ernie", "all"):
        bench = ErnieBenchmark(args.batch_size, args.seq_len, args.lr)
        results["ernie"] = bench.run_comparison(args.steps, args.log_interval)

    if args.model in ("bert", "all"):
        bench = BertBenchmark(args.batch_size, args.seq_len, args.lr)
        results["bert"] = bench.run_comparison(args.steps, args.log_interval)

    if args.model in ("gpt2", "all"):
        bench = GPT2Benchmark(args.batch_size, args.seq_len, args.lr)
        results["gpt2"] = bench.run_comparison(args.steps, args.log_interval)

    # Final summary table
    if len(results) > 1:
        print(f"\n\n{'='*70}")
        print(f"  OVERALL COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Model':<20} {'No-CINN (ms)':<15} {'CINN (ms)':<15} {'Speedup':<10}")
        print(f"  {'-'*60}")
        for name, r in results.items():
            nocinn_avg = np.mean(r["nocinn"]["step_times"][10:])
            cinn_avg = np.mean(r["cinn"]["step_times"][10:])
            print(f"  {name:<20} {nocinn_avg:<15.2f} {cinn_avg:<15.2f} {r['speedup']:<10.2f}x")
        print(f"{'='*70}")
