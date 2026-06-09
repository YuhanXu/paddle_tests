"""Unified CLI entry for the full-train benchmark.

Usage examples:
    python benchmark_train_full.py --model bert
    python benchmark_train_full.py --model all --suite core
    python benchmark_train_full.py --model all --suite full
    python benchmark_train_full.py --model all --suite both
    python benchmark_train_full.py --model bert --datasets sst2,mrpc
    python benchmark_train_full.py --model ernie --max_train_steps 100 --eval_steps 50
    python benchmark_train_full.py --model bert --cinn_mode cinn_only
"""
import sys

# Match path setup of existing benchmark scripts
sys.path = [p for p in sys.path if '/work/Paddle' not in p]
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
sys.path.insert(0, '/work/env3.10/lib/python3.10/site-packages')

import argparse
import json
import os
import time
import traceback

import paddle

from full_train_utils import (
    setup_env, plot_compare, append_summary_csv,
)


# ----- suite definition (mirrors LLM_train_fullstep.md §4.4) -----
SUITE = {
    'core': {
        'bert':  [('sst2',         dict(bs=32, seq=128, epochs=3, lr=2e-5))],
        'ernie': [('chnsenticorp', dict(bs=32, seq=128, epochs=3, lr=5e-5))],
        'gpt2':  [('sst2',         dict(bs=16, seq=128, epochs=3, lr=2e-5))],
        'llama': [('wikitext103',  dict(bs=32, seq=128, epochs=1, lr=3e-4))],
    },
    'full': {
        'bert':  [('sst2',         dict(bs=32, seq=128, epochs=3, lr=2e-5)),
                  ('mrpc',         dict(bs=16, seq=128, epochs=3, lr=2e-5)),
                  ('mnli',         dict(bs=32, seq=128, epochs=3, lr=3e-5))],
        'ernie': [('chnsenticorp', dict(bs=32, seq=128, epochs=3, lr=5e-5)),
                  ('tnews',        dict(bs=32, seq=128, epochs=3, lr=3e-5)),
                  ('lcqmc',        dict(bs=32, seq=128, epochs=3, lr=3e-5))],
        'gpt2':  [('sst2',         dict(bs=16, seq=128, epochs=3, lr=2e-5)),
                  ('mrpc',         dict(bs=16, seq=128, epochs=3, lr=2e-5))],
        'llama': [('wikitext103',  dict(bs=32, seq=128, epochs=3, lr=3e-4)),
                  ('dolly15k',     dict(bs=8,  seq=512, epochs=3, lr=2e-5))],
    },
}

ALL_MODELS = ['bert', 'ernie', 'gpt2', 'llama']


def parse_args():
    p = argparse.ArgumentParser('full-train benchmark')
    p.add_argument('--model', default='all',
                   choices=ALL_MODELS + ['all'])
    p.add_argument('--suite', default='core', choices=['core', 'full', 'both'])
    p.add_argument('--cinn_mode', default='both',
                   choices=['both', 'cinn_only', 'nocinn_only'])
    p.add_argument('--datasets', default=None,
                   help='comma-separated, override suite default')
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--seq_len', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--max_train_steps', type=int, default=-1)
    p.add_argument('--eval_steps', type=int, default=0,
                   help='0 = only at epoch end')
    p.add_argument('--eval_max_batches', type=int, default=200,
                   help='cap dev batches per eval to keep wall time manageable')
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default='./results_full')
    p.add_argument('--resume', default=None)
    p.add_argument('--device', default='gpu:0')
    return p.parse_args()


def resolve_tasks(model_key, args):
    """Return list of (task, hp) tuples based on suite/datasets overrides."""
    if args.datasets:
        names = [s.strip() for s in args.datasets.split(',') if s.strip()]
        # Pull hp defaults from `full` if available, fall back to `core`
        full_map = dict(SUITE['full'].get(model_key, []))
        core_map = dict(SUITE['core'].get(model_key, []))
        out = []
        for n in names:
            hp = full_map.get(n) or core_map.get(n)
            if hp is None:
                # Provide neutral defaults; user supplied a non-suite task.
                hp = dict(bs=32, seq=128, epochs=3, lr=2e-5)
            out.append((n, hp))
        return out

    if args.suite == 'core':
        return list(SUITE['core'].get(model_key, []))
    if args.suite == 'full':
        return list(SUITE['full'].get(model_key, []))
    # both: core first, then full \ core
    core = list(SUITE['core'].get(model_key, []))
    core_names = {t for t, _ in core}
    extras = [(t, hp) for t, hp in SUITE['full'].get(model_key, [])
              if t not in core_names]
    return core + extras


def run_one(model_key, task, hp, cinn_on, args):
    if model_key == 'llama':
        from full_train_clm import run_clm
        return run_clm(task, hp, cinn_on, args)
    from full_train_clf import run_clf
    return run_clf(model_key, task, hp, cinn_on, args)


def cinn_modes(args):
    if args.cinn_mode == 'both':       return [True, False]
    if args.cinn_mode == 'cinn_only':  return [True]
    if args.cinn_mode == 'nocinn_only': return [False]
    raise ValueError(args.cinn_mode)


def main():
    args = parse_args()
    paddle.set_device(args.device)
    setup_env()
    os.makedirs(args.output_dir, exist_ok=True)

    models = ALL_MODELS if args.model == 'all' else [args.model]
    summary_csv = os.path.join(args.output_dir, 'summary_all.csv')
    summary_header = ['model', 'task', 'cinn', 'total_steps',
                      'dev_metric_name', 'dev_metric',
                      'last_step_time_ms', 'wall_time_s']

    for model_key in models:
        tasks = resolve_tasks(model_key, args)
        if not tasks:
            print(f'[skip] no tasks resolved for {model_key}')
            continue
        for task, hp in tasks:
            for cinn_on in cinn_modes(args):
                tag = 'cinn' if cinn_on else 'nocinn'
                t0 = time.time()
                try:
                    summary = run_one(model_key, task, hp, cinn_on, args)
                except Exception as e:
                    traceback.print_exc()
                    print(f'[error] {model_key}/{task}/{tag}: {e}')
                    continue
                wall = time.time() - t0
                metric_name = 'dev_acc' if 'dev_acc' in summary else 'dev_ppl'
                metric_val = summary.get(metric_name)
                append_summary_csv(summary_csv,
                                   [model_key, task, tag, summary['total_steps'],
                                    metric_name, metric_val,
                                    summary.get('last_step_time_ms', ''),
                                    f'{wall:.1f}'],
                                   summary_header)
                print(f'[done] {model_key}/{task}/{tag} '
                      f'{metric_name}={metric_val} wall={wall:.1f}s')

            # Per-task compare plot
            cinn_csv = os.path.join(args.output_dir,
                                     f'{"llama" if model_key=="llama" else model_key}'
                                     f'_{task}_cinn_steps.csv')
            nocinn_csv = os.path.join(args.output_dir,
                                       f'{"llama" if model_key=="llama" else model_key}'
                                       f'_{task}_nocinn_steps.csv')
            png = os.path.join(args.output_dir,
                                f'{"llama" if model_key=="llama" else model_key}'
                                f'_{task}_compare.png')
            metric_col = 'train_loss'
            plot_compare(cinn_csv, nocinn_csv, png,
                         metric_col=metric_col, y_label='train_loss')

    print(f'[all done] summary -> {summary_csv}')


if __name__ == '__main__':
    main()
