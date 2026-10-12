import json

import onnx
import torch

import vocab as V
from dataset import MAX_LEN, ROOM_LABELS, STOCK_TO_ID, TASK_LABELS
from model import IntentClassifier
from tokenizer import PAD, UNK

OUT_DIR = "export"


def main():
    ckpt = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
    model = IntentClassifier(vocab_size=ckpt["vocab_size"], max_len=ckpt["max_len"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # int32, not int64: Sentis's C# Tensor<int> is 32-bit, so the exported
    # graph's declared input type needs to match it directly. model.forward
    # casts to long internally for nn.Embedding, so this is a no-op for
    # correctness and only changes the graph's declared input dtype.
    dummy = torch.zeros((1, ckpt["max_len"]), dtype=torch.int32)
    onnx_path = f"{OUT_DIR}/model.onnx"
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["category", "room", "task", "stock"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "category": {0: "batch"},
            "room": {0: "batch"},
            "task": {0: "batch"},
            "stock": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"wrote {onnx_path} (checked OK)")

    with open("data/token_vocab.json", encoding="utf-8") as f:
        token_vocab = json.load(f)
    # Unity's JsonUtility (the convention already used elsewhere in this
    # project -- see SaveSystem.cs) can't deserialize a JSON object used as
    # a dictionary, only arrays/typed fields. token_vocab's ids are already
    # sequential by construction (build_vocab assigns 0, 1, 2, ... in
    # insertion order), so a plain ordered array round-trips losslessly: the
    # C# side rebuilds the token->id map from each string's array index.
    tokens_by_id = list(token_vocab.keys())
    assert tokens_by_id[0] == PAD and tokens_by_id[1] == UNK
    with open(f"{OUT_DIR}/vocab.json", "w", encoding="utf-8") as f:
        json.dump({"tokens": tokens_by_id, "pad": PAD, "unk": UNK, "maxLen": ckpt["max_len"]}, f, indent=2)

    labels = {
        "categories": V.CATEGORIES,
        "rooms": ROOM_LABELS,
        "tasks": TASK_LABELS,
        "stock": list(STOCK_TO_ID.keys()),
        "noMatch": V.NO_MATCH,
    }
    with open(f"{OUT_DIR}/labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"wrote {OUT_DIR}/vocab.json ({len(token_vocab)} tokens), {OUT_DIR}/labels.json")


if __name__ == "__main__":
    main()
