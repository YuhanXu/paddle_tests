"""Dataset loaders + tokenization + DataLoader builders for full-train benchmark.

Covers 8 datasets:
  clf: SST-2, MRPC, MNLI, ChnSentiCorp, TNEWS, LCQMC
  CLM: WikiText-103
  SFT: Dolly-15K
"""
import sys

# Match path setup of existing benchmark scripts
sys.path = [p for p in sys.path if '/work/Paddle' not in p]
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
sys.path.insert(0, '/work/env3.10/lib/python3.10/site-packages')

from functools import partial
from itertools import chain

import numpy as np
import paddle
from paddle.io import DataLoader
from paddlenlp.data import DataCollatorWithPadding, DataCollatorForLanguageModeling
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer


# ---------- task registry ----------
# fields: kind ('clf'|'clm'|'sft'), num_labels, loader_args, text_keys
TASKS = {
    'sst2':         dict(kind='clf', num_labels=2,  loader=('glue', 'sst-2'),   keys=('sentence',)),
    'mrpc':         dict(kind='clf', num_labels=2,  loader=('glue', 'mrpc'),    keys=('sentence1', 'sentence2')),
    'mnli':         dict(kind='clf', num_labels=3,  loader=('glue', 'mnli'),    keys=('premise', 'hypothesis')),
    'chnsenticorp': dict(kind='clf', num_labels=2,  loader=('chnsenticorp', None), keys=('text',)),
    'tnews':        dict(kind='clf', num_labels=15, loader=('clue', 'tnews'),   keys=('sentence',)),
    'lcqmc':        dict(kind='clf', num_labels=2,  loader=('lcqmc', None),     keys=('query', 'title')),
    'wikitext103':  dict(kind='clm', num_labels=0,  loader=('wikitext', 'wikitext-103-v1'), keys=('text',)),
    'dolly15k':     dict(kind='sft', num_labels=0,  loader=('databricks-dolly-15k', None),  keys=('instruction', 'context', 'response')),
}


def get_task_info(name):
    if name not in TASKS:
        raise ValueError(f'Unknown task: {name}; available: {list(TASKS)}')
    return TASKS[name]


def _safe_load_splits(name, subset, splits):
    args = dict(name=subset) if subset else {}
    return load_dataset(name, splits=splits, **args)


# ---------- clf tokenization ----------
def _make_clf_convert(tokenizer, keys, max_seq_len):
    def convert(example):
        if len(keys) == 1:
            enc = tokenizer(example[keys[0]], max_length=max_seq_len,
                            truncation=True, padding='max_length')
        else:
            enc = tokenizer(example[keys[0]], example[keys[1]],
                            max_length=max_seq_len, truncation=True, padding='max_length')
        # label key heuristics
        for k in ('label', 'labels'):
            if k in example:
                enc['labels'] = int(example[k])
                break
        return enc
    return convert


def build_clf_loaders(task, model_name, batch_size, max_seq_len, shuffle_train=True):
    info = get_task_info(task)
    name, subset = info['loader']
    splits = ['train', 'dev'] if task != 'sst2' else ['train', 'dev']
    train_ds, dev_ds = _safe_load_splits(name, subset, splits)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    conv = _make_clf_convert(tokenizer, info['keys'], max_seq_len)
    train_ds = train_ds.map(conv)
    dev_ds = dev_ds.map(conv)
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=shuffle_train, drop_last=True,
                              collate_fn=collator, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size,
                            shuffle=False, drop_last=False,
                            collate_fn=collator, num_workers=0)
    return train_loader, dev_loader, tokenizer, info['num_labels']


# ---------- CLM (WikiText-103) ----------
def _group_texts(examples, block_size):
    concat = list(chain(*examples['input_ids']))
    total = (len(concat) // block_size) * block_size
    chunks = [concat[i:i + block_size] for i in range(0, total, block_size)]
    return {'input_ids': chunks, 'labels': [c.copy() for c in chunks]}


def build_clm_loaders(model_name, batch_size, block_size, num_proc=1):
    train_ds, dev_ds = _safe_load_splits('wikitext', 'wikitext-103-v1', ['train', 'validation'])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(example):
        return tokenizer(example['text'], add_special_tokens=False)

    def map_and_group(ds):
        # paddlenlp MapDataset: do row-wise; then collate via list comprehension
        ids_list = []
        for ex in ds:
            ids = tokenizer(ex['text'], add_special_tokens=False).get('input_ids', [])
            if ids:
                ids_list.extend(ids)
        # group into block_size
        n = (len(ids_list) // block_size) * block_size
        chunks = [ids_list[i:i + block_size] for i in range(0, n, block_size)]
        return chunks

    train_chunks = map_and_group(train_ds)
    dev_chunks = map_and_group(dev_ds)

    class _ChunkDS(paddle.io.Dataset):
        def __init__(self, chunks):
            self.chunks = chunks
        def __len__(self):
            return len(self.chunks)
        def __getitem__(self, i):
            ids = np.asarray(self.chunks[i], dtype='int64')
            return {'input_ids': ids, 'labels': ids.copy()}

    def _collate(batch):
        input_ids = paddle.to_tensor(np.stack([b['input_ids'] for b in batch]))
        labels = paddle.to_tensor(np.stack([b['labels'] for b in batch]))
        return {'input_ids': input_ids, 'labels': labels}

    train_loader = DataLoader(_ChunkDS(train_chunks), batch_size=batch_size,
                              shuffle=True, drop_last=True, collate_fn=_collate, num_workers=0)
    dev_loader = DataLoader(_ChunkDS(dev_chunks), batch_size=batch_size,
                            shuffle=False, drop_last=False, collate_fn=_collate, num_workers=0)
    return train_loader, dev_loader, tokenizer


# ---------- SFT (Dolly-15K) ----------
def build_sft_loaders(model_name, batch_size, max_seq_len, val_ratio=0.05):
    (full_ds,) = _safe_load_splits('databricks-dolly-15k', None, ['train'])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    PROMPT = ('Below is an instruction that describes a task. '
              'Write a response that appropriately completes the request.\n\n'
              '### Instruction:\n{instruction}\n\n')
    PROMPT_CTX = ('### Input:\n{context}\n\n')
    PROMPT_RESP = '### Response:\n{response}'

    samples = []
    for ex in full_ds:
        prompt = PROMPT.format(instruction=ex.get('instruction', ''))
        if ex.get('context'):
            prompt += PROMPT_CTX.format(context=ex['context'])
        full = prompt + PROMPT_RESP.format(response=ex.get('response', ''))
        ids = tokenizer(full, max_length=max_seq_len, truncation=True,
                        padding='max_length').get('input_ids', [])
        samples.append(ids)

    rng = np.random.RandomState(42)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * val_ratio))
    train_chunks, dev_chunks = samples[n_val:], samples[:n_val]

    class _SFTDS(paddle.io.Dataset):
        def __init__(self, chunks):
            self.chunks = chunks
        def __len__(self):
            return len(self.chunks)
        def __getitem__(self, i):
            ids = np.asarray(self.chunks[i], dtype='int64')
            return {'input_ids': ids, 'labels': ids.copy()}

    def _collate(batch):
        input_ids = paddle.to_tensor(np.stack([b['input_ids'] for b in batch]))
        labels = paddle.to_tensor(np.stack([b['labels'] for b in batch]))
        return {'input_ids': input_ids, 'labels': labels}

    train_loader = DataLoader(_SFTDS(train_chunks), batch_size=batch_size,
                              shuffle=True, drop_last=True, collate_fn=_collate, num_workers=0)
    dev_loader = DataLoader(_SFTDS(dev_chunks), batch_size=batch_size,
                            shuffle=False, drop_last=False, collate_fn=_collate, num_workers=0)
    return train_loader, dev_loader, tokenizer


# ---------- top-level dispatch ----------
def build_loaders(task, model_name, batch_size, seq_len):
    info = get_task_info(task)
    if info['kind'] == 'clf':
        train_loader, dev_loader, tokenizer, num_labels = build_clf_loaders(
            task, model_name, batch_size, seq_len)
        return dict(kind='clf', train=train_loader, dev=dev_loader,
                    tokenizer=tokenizer, num_labels=num_labels)
    elif info['kind'] == 'clm':
        train_loader, dev_loader, tokenizer = build_clm_loaders(
            model_name, batch_size, seq_len)
        return dict(kind='clm', train=train_loader, dev=dev_loader,
                    tokenizer=tokenizer, num_labels=0)
    elif info['kind'] == 'sft':
        train_loader, dev_loader, tokenizer = build_sft_loaders(
            model_name, batch_size, seq_len)
        return dict(kind='sft', train=train_loader, dev=dev_loader,
                    tokenizer=tokenizer, num_labels=0)
    raise ValueError(info['kind'])
