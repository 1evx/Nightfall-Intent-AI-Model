"""
Quick CLI to run the exported ONNX model on arbitrary text -- useful for
sanity-checking the model against real phrasing (including negation-fixed
input, since ApplyKnownFixes runs before this classifier in Unity too)
without going through Unity/Sentis. Not used by training or export; this
mirrors what SurvivorTrainedIntentClassifier.cs does in C#.
"""
import json
import sys

import numpy as np
import onnxruntime as ort

from tokenizer import encode, tokenize

with open("export/vocab.json", encoding="utf-8") as f:
    vocab_data = json.load(f)
with open("export/labels.json", encoding="utf-8") as f:
    labels = json.load(f)

# vocab.json's "tokens" is an id-ordered array (JsonUtility-friendly on the
# Unity side, see export.py) -- rebuild the token->id dict locally.
token_to_id = {tok: i for i, tok in enumerate(vocab_data["tokens"])}

session = ort.InferenceSession("export/model.onnx")


def predict(text):
    ids = encode(tokenize(text), token_to_id, vocab_data["maxLen"])
    input_arr = np.array([ids], dtype=np.int32)
    outputs = session.run(None, {"input_ids": input_arr})
    names = ["category", "room", "task", "stock"]
    result = {}
    for name, out in zip(names, outputs):
        idx = int(np.argmax(out[0]))
        label_list = {"category": labels["categories"], "room": labels["rooms"],
                      "task": labels["tasks"], "stock": labels["stock"]}[name]
        result[name] = label_list[idx]
    return result


SAMPLES = [
    "follow me", "stay here", "resume your routine", "go to the kitchen",
    "give her some food", "give me back the pills", "grab some medicine yourself",
    "how much medicine do we have", "what's in stock", "where are you",
    "how are you doing", "what do you suggest", "go work the lumberyard",
    "can you take me to the library", "nice weather today", "arrest roommate",
    "go back to your usual routine", "central hall",
]

if __name__ == "__main__":
    samples = sys.argv[1:] if len(sys.argv) > 1 else SAMPLES
    for text in samples:
        r = predict(text)
        category = r["category"]
        slot = ""
        if category == "Room":
            slot = f" room={r['room']}"
        elif category == "Task":
            slot = f" task={r['task']}"
        elif category in ("Give", "Return", "ResourceOne"):
            slot = f" stock={r['stock']}"
        print(f"{text!r:55} -> {category}{slot}")
