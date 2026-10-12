"""
Preprocessing shared by training and export. This exact logic (lowercase,
then split on runs of word-characters/apostrophes) gets ported to C# in
SurvivorTrainedIntentClassifier.Tokenize -- keep the two in lockstep, since
any divergence here silently breaks the trained model's real-world accuracy
without showing up as a training-time bug.
"""
import re

PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(token_lists, min_freq=1):
    from collections import Counter
    counts = Counter(tok for toks in token_lists for tok in toks)
    vocab = {PAD: 0, UNK: 1}
    for tok, freq in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(tokens, vocab, max_len):
    ids = [vocab.get(t, vocab[UNK]) for t in tokens[:max_len]]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids
