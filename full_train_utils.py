"""Shared utilities for full-train benchmark: CINN wrap / scheduler / CSV / plot / eval."""
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import paddle


# ---------- env / seed ----------
def setup_env():
    os.environ.setdefault('FLAGS_prim_all', 'true')
    paddle.set_flags({'FLAGS_print_ir': False, 'FLAGS_deny_cinn_ops': ''})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


# ---------- CINN wrap ----------
def to_cinn_net(net, **kwargs):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = True
    return paddle.jit.to_static(net, build_strategy=build_strategy,
                                full_graph=True, **kwargs)


def maybe_wrap_cinn(net, cinn_on, **kwargs):
    return to_cinn_net(net, **kwargs) if cinn_on else net


# ---------- scheduler ----------
def make_lr_and_optimizer(model, total_steps, lr, warmup_ratio=0.1, weight_decay=0.01):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    sched = paddle.optimizer.lr.LinearWarmup(
        learning_rate=paddle.optimizer.lr.PolynomialDecay(
            learning_rate=lr,
            decay_steps=max(1, total_steps - warmup_steps),
            end_lr=0.0, power=1.0),
        warmup_steps=warmup_steps,
        start_lr=0.0,
        end_lr=lr,
    )
    decay_params = [p.name for n, p in model.named_parameters()
                    if not any(nd in n for nd in ['bias', 'norm', 'LayerNorm'])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=sched,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
    )
    return sched, optimizer


# ---------- CSV ----------
def open_csv_writer(path, header):
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(header)
    return f, w


def append_csv(writer, row):
    writer.writerow(row)


# ---------- plotting ----------
def plot_compare(cinn_csv, nocinn_csv, out_png, metric_col='loss', y_label='loss'):
    """Two-subplot compare. Left: metric vs step. Right: step_time_ms vs step."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[plot] matplotlib not available, skip')
        return

    def read(path):
        steps, vals, times = [], [], []
        if not os.path.exists(path):
            return steps, vals, times
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    steps.append(int(r['step']))
                    vals.append(float(r.get(metric_col, 'nan')))
                    times.append(float(r.get('step_time_ms', 'nan')))
                except (ValueError, KeyError):
                    continue
        return steps, vals, times

    s1, v1, t1 = read(cinn_csv)
    s2, v2, t2 = read(nocinn_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if s1: axes[0].plot(s1, v1, label='CINN')
    if s2: axes[0].plot(s2, v2, label='No-CINN')
    axes[0].set_xlabel('step'); axes[0].set_ylabel(y_label)
    axes[0].set_title(f'{y_label} curve'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if s1: axes[1].plot(s1, t1, label='CINN')
    if s2: axes[1].plot(s2, t2, label='No-CINN')
    axes[1].set_xlabel('step'); axes[1].set_ylabel('step_time_ms')
    axes[1].set_title('step time'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'[plot] wrote {out_png}')


# ---------- evaluation ----------
@paddle.no_grad()
def eval_clf(model, dev_loader, max_batches=None):
    model.eval()
    total, correct, loss_sum, n_batches = 0, 0, 0.0, 0
    for i, batch in enumerate(dev_loader):
        if max_batches is not None and i >= max_batches:
            break
        out = model(**{k: v for k, v in batch.items() if k != 'labels'},
                    labels=batch['labels'])
        loss = out[0] if isinstance(out, tuple) else out.loss
        logits = out[1] if isinstance(out, tuple) else out.logits
        preds = paddle.argmax(logits, axis=-1)
        correct += int((preds == batch['labels']).astype('int64').sum().item())
        total += int(batch['labels'].shape[0])
        loss_sum += float(loss.item())
        n_batches += 1
    model.train()
    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, n_batches)
    return dict(dev_acc=acc, dev_loss=avg_loss)


@paddle.no_grad()
def eval_clm(model, dev_loader, max_batches=None):
    model.eval()
    loss_sum, n_batches = 0.0, 0
    for i, batch in enumerate(dev_loader):
        if max_batches is not None and i >= max_batches:
            break
        out = model(input_ids=batch['input_ids'], labels=batch['labels'])
        loss = out[0] if isinstance(out, tuple) else out.loss
        loss_sum += float(loss.item())
        n_batches += 1
    model.train()
    avg = loss_sum / max(1, n_batches)
    return dict(dev_loss=avg, dev_ppl=math.exp(min(avg, 20)))


# ---------- summary ----------
def write_summary_json(path, info_dict):
    with open(path, 'w') as f:
        json.dump(info_dict, f, indent=2)


def append_summary_csv(path, row, header):
    is_new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)


# ---------- timing helper ----------
class StepTimer:
    def __init__(self):
        self.start = None
    def tick(self):
        paddle.device.synchronize()
        self.start = time.time()
    def tock(self):
        paddle.device.synchronize()
        return (time.time() - self.start) * 1000.0
