"""Residual stacked GRU with candidate-aware attention and learned GBM inputs."""

import math
import torch
from torch import nn


class GRUProductModel(nn.Module):
    def __init__(
        self, event_count, product_size, classes, width=256, layers=4, dropout=0.15
    ):
        super().__init__()
        self.config = dict(
            event_count=event_count,
            product_size=product_size,
            classes=list(classes),
            width=width,
            layers=layers,
            dropout=dropout,
        )
        self.events = nn.Embedding(event_count + 1, width, padding_idx=0)
        self.products = nn.Embedding(product_size, width, padding_idx=0)
        self.gaps = nn.Embedding(27, width, padding_idx=0)
        self.hours = nn.Embedding(25, width, padding_idx=0)
        self.weekdays = nn.Embedding(8, width, padding_idx=0)
        self.boundaries = nn.Embedding(3, width, padding_idx=0)
        self.input_norm = nn.LayerNorm(width)
        self.grus = nn.ModuleList(
            [nn.GRU(width, width, batch_first=True) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.key = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.relation = nn.Embedding(event_count + 1, 3, padding_idx=0)
        self.project = nn.Sequential(nn.Linear(width, 32), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(32 * 4 + 12, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.bias = nn.Parameter(torch.zeros(len(classes)))
        self.register_buffer(
            "class_tokens", torch.tensor(classes, dtype=torch.long) + 3
        )

    def forward(
        self,
        events,
        products,
        gaps,
        hours,
        weekdays,
        boundaries,
        lengths,
        features,
        return_features=False,
    ):
        safe = torch.where(
            (products >= 0) & (products < self.products.num_embeddings), products, 2
        )
        x = (
            self.events(events)
            + self.products(safe)
            + self.gaps(gaps)
            + self.hours(hours)
            + self.weekdays(weekdays)
            + self.boundaries(boundaries)
        )
        x = self.dropout(self.input_norm(x))
        for gru, norm in zip(self.grus, self.norms):
            update, _ = gru(x)
            x = norm(x + self.dropout(update))
        # Forward-only recurrence: right padding cannot affect genuine positions.
        pad = torch.arange(x.shape[1], device=x.device)[None, :] >= lengths[:, None]
        last = x[torch.arange(len(x), device=x.device), lengths - 1]
        candidate = self.products(self.class_tokens)
        query = self.query(candidate)[None, :, :] + last[:, None, :]
        attention = torch.bmm(query, self.key(x).transpose(1, 2)) / math.sqrt(
            x.shape[-1]
        )
        same = (products[:, None, :] == self.class_tokens[None, :, None]) & ~pad[
            :, None, :
        ]
        relation = self.relation(events)
        attention = attention + torch.where(
            same,
            relation[:, None, :, 2],
            torch.where(
                products[:, None, :] == 1,
                relation[:, None, :, 0],
                relation[:, None, :, 1],
            ),
        )
        attended = torch.bmm(
            attention.masked_fill(pad[:, None, :], -1e4).softmax(-1), x
        )
        weights = same.to(x.dtype)
        related = torch.bmm(weights / weights.sum(-1, keepdim=True).clamp_min(1), x)
        user = self.project(last)
        context = self.project(attended)
        n, c = features.shape[:2]
        merged = torch.cat(
            [
                user[:, None, :].expand(-1, c, -1),
                context,
                self.project(related),
                self.project(candidate)[None, :, :].expand(n, -1, -1),
                features,
            ],
            dim=-1,
        )
        scores = self.head(merged).squeeze(-1) + self.bias
        if return_features:
            return scores, user, context
        return scores

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
