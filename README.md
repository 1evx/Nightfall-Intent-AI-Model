# Nightfall Intent AI Model

A compact, multi-head neural intent classifier for the survivor command system in **Shelter In — Nightfall**. It converts a natural-language player command into an intent category and, when relevant, a room, task, or stock-item slot.

The model is designed for local CPU inference in Unity through Sentis. Training uses PyTorch, and the deployable model is exported to ONNX.

## What it predicts

One forward pass produces four outputs:

- **Category:** one of 13 command intents, including `Follow`, `Stay`, `Room`, `Task`, `Give`, `Return`, and `None`
- **Room:** one of 18 known rooms, plus `NoMatch`
- **Task:** one of 4 survivor tasks, plus `NoMatch`
- **Stock:** `None`, `Food`, `Pill`, or `Medicine`

The architecture is intentionally small:

```text
tokens → embedding → trigram Conv1d → masked mean/max pooling
       → shared hidden layer → category / room / task / stock heads
```

The current model contains 67,785 parameters. On the held-out synthetic test set, the documented run achieved **92.9% category accuracy** and **100% conditional slot accuracy** for room, task, and stock predictions. These figures measure performance on template-generated data, not real-world player input.

## Repository layout

```text
data_gen.py          Generate and split the synthetic command dataset
vocab.py             Intent, room, task, and stock vocabulary
tokenizer.py         Shared tokenization and encoding rules
dataset.py           PyTorch dataset and label mappings
model.py             Multi-head convolutional classifier
train.py             Training, early stopping, and test evaluation
export.py            Export a checkpoint to ONNX and JSON metadata
predict.py           Run local predictions with ONNX Runtime
data/                Generated train, validation, and test data
export/              Unity-ready ONNX model, vocabulary, and labels
report/              Detailed implementation and evaluation notes
```

## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- ONNX and ONNX Script
- ONNX Runtime

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run all commands below from the repository root because the scripts use project-relative paths.

## Quick start

The repository includes an exported model, so you can try inference immediately:

```bash
python predict.py "follow me" "go work the lumberyard" "how much medicine do we have"
```

With no arguments, `predict.py` evaluates its built-in sample commands:

```bash
python predict.py
```

## Rebuild the model

Generate the synthetic dataset:

```bash
python data_gen.py
```

Train and evaluate the classifier:

```bash
python train.py
```

Training writes a local `checkpoint.pt`. It is excluded from Git because it is reproducible and is only an intermediate artifact.

Export the trained checkpoint:

```bash
python export.py
```

This produces the deployment files in `export/`:

- `model.onnx` — the trained model
- `vocab.json` — ordered tokens and tokenizer metadata
- `labels.json` — output label mappings

These three files are intentionally included in the repository so the classifier can be integrated into Unity without retraining it.

## Unity integration

The ONNX graph accepts an `int32` tensor named `input_ids` with shape `(batch, 24)` and returns four logits tensors named `category`, `room`, `task`, and `stock`. Tokenization must match `tokenizer.py`: lowercase the input, extract words with `[a-z0-9]+(?:'[a-z0-9]+)?`, map unknown words to `<unk>`, then pad or truncate to 24 tokens.

The exported vocabulary is an ID-ordered array so it can be loaded with Unity's `JsonUtility`. The consuming Unity component should rebuild the token-to-ID mapping from each array index.

## Dataset and limitations

The included dataset contains 3,702 synthetic examples split by phrase family:

| Split | Examples |
|---|---:|
| Train | 2,956 |
| Validation | 366 |
| Test | 380 |

Family-level splitting keeps augmented versions of the same sentence template together, reducing leakage between splits. However, synthetic templates do not fully represent real player language, typos, speech-recognition errors, or new rooms and tasks. Treat this model as a project-specific command classifier rather than a general natural-language understanding model.

