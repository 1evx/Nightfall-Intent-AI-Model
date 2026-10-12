import json

import torch
from torch.utils.data import Dataset

import vocab as V
from tokenizer import build_vocab, encode, tokenize

MAX_LEN = 24

CATEGORY_TO_ID = {c: i for i, c in enumerate(V.CATEGORIES)}
ROOM_LABELS = V.ROOMS + [V.NO_MATCH]
ROOM_TO_ID = {r: i for i, r in enumerate(ROOM_LABELS)}
TASK_LABELS = V.TASK_LABELS + [V.NO_MATCH]
TASK_TO_ID = {t: i for i, t in enumerate(TASK_LABELS)}
STOCK_TO_ID = {s: i for i, s in enumerate(V.STOCK_TYPES)}

IGNORE_INDEX = -100


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class IntentDataset(Dataset):
    def __init__(self, examples, token_vocab, max_len=MAX_LEN):
        self.examples = examples
        self.token_vocab = token_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids = encode(tokenize(ex["text"]), self.token_vocab, self.max_len)

        category = CATEGORY_TO_ID[ex["category"]]
        room = ROOM_TO_ID[ex["room"]] if ex["room"] else IGNORE_INDEX
        task = TASK_TO_ID[ex["task"]] if ex["task"] else IGNORE_INDEX
        stock = STOCK_TO_ID[ex["stock"]] if ex["stock"] else IGNORE_INDEX

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "category": torch.tensor(category, dtype=torch.long),
            "room": torch.tensor(room, dtype=torch.long),
            "task": torch.tensor(task, dtype=torch.long),
            "stock": torch.tensor(stock, dtype=torch.long),
        }


def build_token_vocab_from_train():
    train = load_jsonl("data/train.jsonl")
    return build_vocab([tokenize(ex["text"]) for ex in train])
