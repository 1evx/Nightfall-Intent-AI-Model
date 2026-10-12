import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (CATEGORY_TO_ID, IGNORE_INDEX, IntentDataset, MAX_LEN,
                      ROOM_LABELS, STOCK_TO_ID, TASK_LABELS,
                      build_token_vocab_from_train, load_jsonl)
from model import IntentClassifier
from tokenizer import PAD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEADS = ["category", "room", "task", "stock"]


def make_loader(examples, token_vocab, batch_size, shuffle):
    ds = IntentDataset(examples, token_vocab, max_len=MAX_LEN)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    total_loss = 0.0
    correct = {h: 0 for h in HEADS}
    total = {h: 0 for h in HEADS}

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        targets = {h: batch[h].to(DEVICE) for h in HEADS}

        with torch.set_grad_enabled(train_mode):
            outputs = model(input_ids)
            head_losses = [loss_fn(outputs[h], targets[h]) for h in HEADS if (targets[h] != IGNORE_INDEX).any()]
            loss = sum(head_losses)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        for h in HEADS:
            mask = targets[h] != IGNORE_INDEX
            if mask.any():
                preds = outputs[h].argmax(dim=-1)
                correct[h] += (preds[mask] == targets[h][mask]).sum().item()
                total[h] += mask.sum().item()

    acc = {h: (correct[h] / total[h] if total[h] else float("nan")) for h in HEADS}
    return total_loss / len(loader.dataset), acc


def confusion_report(model, loader, id_to_category):
    model.eval()
    n = len(id_to_category)
    confusion = [[0] * n for _ in range(n)]
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            targets = batch["category"].to(DEVICE)
            preds = model(input_ids)["category"].argmax(dim=-1)
            for t, p in zip(targets.tolist(), preds.tolist()):
                confusion[t][p] += 1

    print(f"{'':14}" + "".join(f"{id_to_category[j][:6]:>7}" for j in range(n)))
    for i in range(n):
        row = "".join(f"{confusion[i][j]:>7}" for j in range(n))
        print(f"{id_to_category[i]:14}{row}")


def main():
    token_vocab = build_token_vocab_from_train()
    with open("data/token_vocab.json", "w", encoding="utf-8") as f:
        json.dump(token_vocab, f, indent=2)
    print(f"token vocab size: {len(token_vocab)}")

    train_examples = load_jsonl("data/train.jsonl")
    val_examples = load_jsonl("data/val.jsonl")
    test_examples = load_jsonl("data/test.jsonl")

    train_loader = make_loader(train_examples, token_vocab, batch_size=32, shuffle=True)
    val_loader = make_loader(val_examples, token_vocab, batch_size=64, shuffle=False)
    test_loader = make_loader(test_examples, token_vocab, batch_size=64, shuffle=False)

    model = IntentClassifier(vocab_size=len(token_vocab), max_len=MAX_LEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = -1.0
    best_state = None
    patience, bad_epochs = 8, 0
    max_epochs = 60

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader)
        print(f"epoch {epoch:2d}  train_loss={train_loss:.4f} cat_acc={train_acc['category']:.3f}"
              f"  val_loss={val_loss:.4f} val_cat_acc={val_acc['category']:.3f}"
              f"  val_room={val_acc['room']:.3f} val_task={val_acc['task']:.3f} val_stock={val_acc['stock']:.3f}")

        if val_acc["category"] > best_val_acc:
            best_val_acc = val_acc["category"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (best val_cat_acc={best_val_acc:.3f})")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "vocab_size": len(token_vocab), "max_len": MAX_LEN},
               "checkpoint.pt")

    print("\n--- test set ---")
    test_loss, test_acc = run_epoch(model, test_loader)
    print(f"test_loss={test_loss:.4f} " + " ".join(f"{h}_acc={test_acc[h]:.3f}" for h in HEADS))

    id_to_category = {i: c for c, i in CATEGORY_TO_ID.items()}
    print("\ncategory confusion matrix (test set, rows=true, cols=pred):")
    confusion_report(model, test_loader, id_to_category)


if __name__ == "__main__":
    main()
