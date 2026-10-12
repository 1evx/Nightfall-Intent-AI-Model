import torch
import torch.nn as nn

import vocab as V

NUM_CATEGORIES = len(V.CATEGORIES)
NUM_ROOMS = len(V.ROOMS) + 1  # + NoMatch
NUM_TASKS = len(V.TASK_LABELS) + 1  # + NoMatch
NUM_STOCK = len(V.STOCK_TYPES)


class IntentClassifier(nn.Module):
    """Embedding -> a same-length trigram Conv1d -> mean+max pool -> shared
    hidden layer -> four independent classification heads (category, room,
    task, stock).

    Plain per-token mean/max pooling can't tell "back" from "give back":
    several of this grammar's real trigger phrases are two/three-word units
    (Return's "give back"/"take back", Give's "central hall", the
    "how many"/"how much" quantity trigger) that mean something different
    from their words in isolation (Resume's "go back to your routine" has no
    Return meaning at all). The Conv1d slides a 3-token window across the
    sequence before pooling so the network can learn those combinations as a
    unit instead of just per-word presence. Small and CPU-friendly by
    design: this runs once per player command inside Unity via Sentis, not
    batched, not on a schedule.
    """

    def __init__(self, vocab_size, emb_dim=64, conv_channels=96, hidden_dim=128, max_len=24):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # kernel_size=3, padding=1 keeps the sequence length unchanged, so
        # the token mask below stays valid for pooling the conv output too.
        self.conv = nn.Conv1d(emb_dim, conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Train accuracy saturates at 100% within a handful of epochs on this
        # small dataset while val/test lag well behind -- textbook
        # overfitting to the exact template surface forms, not the intent.
        # Dropout on the pooled features and the hidden layer is the direct
        # countermeasure.
        self.dropout = nn.Dropout(0.3)
        self.hidden = nn.Sequential(nn.Linear(conv_channels * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.category_head = nn.Linear(hidden_dim, NUM_CATEGORIES)
        self.room_head = nn.Linear(hidden_dim, NUM_ROOMS)
        self.task_head = nn.Linear(hidden_dim, NUM_TASKS)
        self.stock_head = nn.Linear(hidden_dim, NUM_STOCK)

    def forward(self, input_ids):
        # input_ids: (batch, max_len), 0 = pad. Cast (not assumed) so the
        # exported ONNX graph can declare an int32 input -- matching Sentis's
        # C# Tensor<int> directly -- while nn.Embedding still gets the int64
        # indices it requires internally.
        input_ids = input_ids.long()
        mask = (input_ids != 0).float()  # (batch, max_len)
        emb = self.embedding(input_ids)  # (batch, max_len, emb_dim)

        # Zero out padded positions before the conv so pad embeddings don't
        # leak into a real token's 3-wide window at the sequence boundary.
        emb = emb * mask.unsqueeze(-1)
        conv_out = self.relu(self.conv(emb.transpose(1, 2)))  # (batch, conv_channels, max_len)
        conv_out = conv_out.transpose(1, 2)  # (batch, max_len, conv_channels)

        mask_3d = mask.unsqueeze(-1)  # (batch, max_len, 1)
        summed = (conv_out * mask_3d).sum(dim=1)
        count = mask_3d.sum(dim=1).clamp(min=1.0)
        mean_pooled = summed / count

        neg_inf = torch.finfo(conv_out.dtype).min
        masked_for_max = conv_out.masked_fill(mask_3d == 0, neg_inf)
        max_pooled = masked_for_max.max(dim=1).values

        pooled = self.dropout(torch.cat([mean_pooled, max_pooled], dim=-1))

        h = self.hidden(pooled)
        return {
            "category": self.category_head(h),
            "room": self.room_head(h),
            "task": self.task_head(h),
            "stock": self.stock_head(h),
        }
