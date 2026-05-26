import sys
# 确保加载带 CUDA 的 paddle（排除源码编译路径），同时用 env3.10 的 paddlenlp
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

cinn_denied_ops = [
    # "arg_max",
    # "argmax",
    # "bitwise_and",
    # "concat",
    # "cumsum",
    # "gather",
    # "gather_nd",
    # "lookup_table_v2",
    # "reduce_sum",
    # "sum",
    # "reduce_max",
    # "max",
    # "slice",
    # "strided_slice",
    # "roll",
    # "tile",
    # "transpose2",
    # "transpose",
    # "range",
    # "arange",
    # "fill_constant",
]

paddle.set_device('gpu:0')

paddle.set_flags({
    "FLAGS_print_ir": False,
    "FLAGS_deny_cinn_ops": ";".join(cinn_denied_ops),
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


def benchmark_train(net, input_ids, labels, optimizer, repeat=10, warmup=3):
    """训练性能测试：前向 + loss + 反向 + 优化器更新"""
    net.train()

    # warm up
    for i in range(warmup):
        outputs = net(input_ids=input_ids, labels=labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
    paddle.device.synchronize()

    # time
    t = []
    for i in range(repeat):
        t1 = time.time()
        outputs = net(input_ids=input_ids, labels=labels)
        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        paddle.device.synchronize()
        t2 = time.time()
        t.append((t2 - t1) * 1000)

    avg = np.mean(t)
    std = np.std(t)
    print(f"--[benchmark_train] Run for {repeat} times, avg latency: {avg:.2f} ms, std: {std:.2f} ms")
    print(f"--[benchmark_train] throughput: {1000.0 / avg:.2f} steps/s")
    return t


class TestLlama2Train:
    def __init__(self, batch_size=1, seq_len=128, model_name='meta-llama/Llama-2-7b-chat',
                 learning_rate=1e-5, use_fp16=False):
        """
        Llama2 训练性能测试

        Args:
            batch_size: 批大小
            seq_len: 序列长度
            model_name: 模型名称/路径
            learning_rate: 学习率
            use_fp16: 是否使用混合精度训练
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.use_fp16 = use_fp16
        self.vocab_size = 32000  # Llama2 默认词表大小

        self.net = None
        self.cinn_net = None
        self.optimizer = None
        self.scaler = None

        # 初始化输入数据
        self.input_ids = self.init_input()
        self.labels = self.init_labels()

    def init_input(self):
        """生成随机 input_ids"""
        return paddle.randint(0, self.vocab_size, [self.batch_size, self.seq_len])

    def init_labels(self):
        """
        生成训练标签。对于因果语言模型，labels 通常是 input_ids 左移一位。
        这里直接用随机标签模拟，-100 表示忽略位置。
        """
        labels = paddle.randint(0, self.vocab_size, [self.batch_size, self.seq_len])
        return labels

    def init_model(self):
        """加载 Llama2 模型"""
        print(f"--[init_model] Loading {self.model_name} ...")
        model = paddlenlp.transformers.LlamaForCausalLM.from_pretrained(self.model_name)
        print(f"--[init_model] Model loaded. Parameters: {sum(p.numel().item() for p in model.parameters()) / 1e9:.2f}B")
        return model

    def init_optimizer(self, net):
        """初始化优化器"""
        # 使用 AdamW，适合大模型微调
        optimizer = paddle.optimizer.AdamW(
            parameters=net.parameters(),
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
        )
        return optimizer

    def get_net(self, use_cinn):
        """获取模型（动态图 or CINN 编译）"""
        if use_cinn:
            if self.cinn_net is None:
                if self.net is None:
                    self.net = self.init_model()
                self.cinn_net = to_cinn_net(self.net)
                self.net = None
            return self.cinn_net
        else:
            if self.net is None:
                self.net = self.init_model()
            return self.net

    def benchmark(self, use_cinn, repeat=10, warmup=3):
        """运行训练 benchmark"""
        mode_str = "cinn" if use_cinn else "nocinn"
        print(f"\n{'='*60}")
        print(f"--[benchmark_train] Llama2 Training Benchmark ({mode_str})")
        print(f"--[config] batch_size={self.batch_size}, seq_len={self.seq_len}, fp16={self.use_fp16}")
        print(f"{'='*60}")

        net = self.get_net(use_cinn)
        net.train()
        optimizer = self.init_optimizer(net)

        if self.use_fp16:
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            self._benchmark_fp16(net, optimizer, scaler, repeat, warmup)
        else:
            benchmark_train(net, self.input_ids, self.labels, optimizer, repeat, warmup)

    def _benchmark_fp16(self, net, optimizer, scaler, repeat=10, warmup=3):
        """混合精度训练 benchmark"""
        net.train()

        # warm up
        for i in range(warmup):
            with paddle.amp.auto_cast(level='O1'):
                outputs = net(input_ids=self.input_ids, labels=self.labels)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()
        paddle.device.synchronize()

        # time
        t = []
        for i in range(repeat):
            t1 = time.time()
            with paddle.amp.auto_cast(level='O1'):
                outputs = net(input_ids=self.input_ids, labels=self.labels)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()
            paddle.device.synchronize()
            t2 = time.time()
            t.append((t2 - t1) * 1000)

        avg = np.mean(t)
        std = np.std(t)
        print(f"--[benchmark_train_fp16] Run for {repeat} times, avg latency: {avg:.2f} ms, std: {std:.2f} ms")
        print(f"--[benchmark_train_fp16] throughput: {1000.0 / avg:.2f} steps/s")

    def check_train_step(self, use_cinn):
        """验证单步训练是否正常（loss 是否有效下降）"""
        mode_str = "cinn" if use_cinn else "nocinn"
        print(f"\n--[check_train_step] Verifying training step ({mode_str}) ...")

        net = self.get_net(use_cinn)
        net.train()
        optimizer = self.init_optimizer(net)

        losses = []
        for step in range(5):
            outputs = net(input_ids=self.input_ids, labels=self.labels)
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses.append(loss.item())
            print(f"  step {step}: loss = {loss.item():.4f}")

        print(f"--[check_train_step] Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
        if losses[-1] < losses[0]:
            print("--[check_train_step] PASS: loss is decreasing.")
        else:
            print("--[check_train_step] WARN: loss not decreasing (may be normal for random data).")

    def train_n_steps(self, use_cinn, total_steps=500, log_interval=10, save_log=True):
        """
        训练 N 步并收集 loss 信息

        Args:
            use_cinn: 是否使用 CINN
            total_steps: 总训练步数
            log_interval: 每隔多少步打印一次 loss
            save_log: 是否将 loss 日志保存到文件
        """
        mode_str = "cinn" if use_cinn else "nocinn"
        print(f"\n{'='*60}")
        print(f"--[train_n_steps] Llama2 Training {total_steps} steps ({mode_str})")
        print(f"--[config] batch_size={self.batch_size}, seq_len={self.seq_len}, "
              f"fp16={self.use_fp16}, lr={self.learning_rate}")
        print(f"{'='*60}")

        net = self.get_net(use_cinn)
        net.train()
        optimizer = self.init_optimizer(net)

        if self.use_fp16:
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        losses = []
        step_times = []
        total_start = time.time()

        for step in range(total_steps):
            t1 = time.time()

            if self.use_fp16:
                with paddle.amp.auto_cast(level='O1'):
                    outputs = net(input_ids=self.input_ids, labels=self.labels)
                    loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(input_ids=self.input_ids, labels=self.labels)
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
                print(f"  step [{step+1:>4d}/{total_steps}] "
                      f"loss: {loss_val:.6f} | "
                      f"avg_time: {avg_time:.2f} ms/step | "
                      f"throughput: {1000.0/avg_time:.2f} steps/s")

        total_time = time.time() - total_start

        # 统计汇总
        print(f"\n{'='*60}")
        print(f"--[Summary] {mode_str} | {total_steps} steps completed in {total_time:.1f}s")
        print(f"--[Loss] initial: {losses[0]:.6f} | final: {losses[-1]:.6f} | "
              f"min: {min(losses):.6f} | max: {max(losses):.6f}")
        print(f"--[Time] avg: {np.mean(step_times):.2f} ms | "
              f"p50: {np.percentile(step_times, 50):.2f} ms | "
              f"p95: {np.percentile(step_times, 95):.2f} ms | "
              f"p99: {np.percentile(step_times, 99):.2f} ms")
        print(f"--[Throughput] avg: {1000.0/np.mean(step_times):.2f} steps/s | "
              f"tokens/s: {self.batch_size * self.seq_len * 1000.0 / np.mean(step_times):.0f}")
        print(f"{'='*60}")

        # 保存日志
        if save_log:
            log_file = f"llama2_train_{mode_str}_bs{self.batch_size}_seq{self.seq_len}_{total_steps}steps.csv"
            with open(log_file, 'w') as f:
                f.write("step,loss,step_time_ms\n")
                for i in range(len(losses)):
                    f.write(f"{i+1},{losses[i]:.6f},{step_times[i]:.2f}\n")
            print(f"--[Save] Loss log saved to: {log_file}")

        return {"losses": losses, "step_times": step_times, "total_time": total_time}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Llama2 Training Benchmark")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat",
                        help="Model name or path")
    parser.add_argument("--repeat", type=int, default=10, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (AMP O1)")
    parser.add_argument("--use_cinn", action="store_true", help="Use CINN compilation")
    parser.add_argument("--no_cinn", action="store_true", help="Run without CINN (dynamic graph)")
    parser.add_argument("--check", action="store_true", help="Run training correctness check")
    parser.add_argument("--train", action="store_true", help="Run N-step training with loss collection")
    parser.add_argument("--steps", type=int, default=500, help="Total training steps (for --train mode)")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps (for --train mode)")
    args = parser.parse_args()

    print(f"Paddle version: {paddle.__version__}")
    print(f"Device: {paddle.get_device()}")
    print(paddle.get_flags("FLAGS_deny_cinn_ops"))

    model = TestLlama2Train(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        model_name=args.model_name,
        learning_rate=args.lr,
        use_fp16=args.fp16,
    )

    if args.check:
        # 验证训练是否正常
        if not args.use_cinn:
            model.check_train_step(use_cinn=False)
        if not args.no_cinn:
            model.check_train_step(use_cinn=True)
    elif args.train:
        # 跑 N 步训练，收集 loss 信息
        if not args.use_cinn:
            model.train_n_steps(use_cinn=False, total_steps=args.steps, log_interval=args.log_interval)
        if not args.no_cinn:
            model.train_n_steps(use_cinn=True, total_steps=args.steps, log_interval=args.log_interval)
    else:
        # 运行 benchmark（短时间性能测试）
        if not args.use_cinn:
            # 默认先跑动态图
            model.benchmark(use_cinn=False, repeat=args.repeat, warmup=args.warmup)

        if not args.no_cinn:
            # 再跑 CINN
            model.benchmark(use_cinn=True, repeat=args.repeat, warmup=args.warmup)
