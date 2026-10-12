"""Tiny attention-exposing transformer and student-aware distillation losses."""
import torch
from torch import nn
from torch.nn import functional as F


def checkpoint_indices(depth: int, points: int) -> list[int]:
    if depth < 1 or points < 1 or points > depth:
        raise ValueError("Require 1 <= points <= depth")
    if points == 1:
        return [depth - 1]
    return [(i * (depth - 1)) // (points - 1) for i in range(points)]


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, x):
        seq = x.size(1)
        # True positions are disallowed, matching causal decoding.
        causal = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
        y, weights = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                               attn_mask=causal, need_weights=True, average_attn_weights=False)
        x = x + y
        return x + self.ff(self.norm2(x)), weights


class TinyTransformer(nn.Module):
    def __init__(self, vocab=32, width=48, depth=7, heads=4, classes=2, max_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(max_len, width)
        self.blocks = nn.ModuleList([AttentionBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)

    def forward(self, tokens):
        batch, length = tokens.shape
        x = self.embed(tokens) + self.pos(torch.arange(length, device=tokens.device))[None]
        maps = []
        for block in self.blocks:
            x, attn = block(x)
            maps.append(attn)
        return self.head(self.norm(x[:, -1])), maps


class MaskGenerator(nn.Module):
    """Shared edge scorer with checkpoint-specific biases; soft masks remain differentiable."""
    def __init__(self, points: int, hidden=24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.bias = nn.Parameter(torch.zeros(points))

    def forward(self, teacher_attention, student_attention, point: int):
        # Work on per-edge head means to accommodate differing teacher/student head counts.
        t = teacher_attention.detach().mean(dim=1)
        s = student_attention.detach().mean(dim=1)
        error = (t - s).abs()
        features = torch.stack((t, s, error), dim=-1)
        mask = torch.sigmoid(self.net(features).squeeze(-1) + self.bias[point])
        valid = torch.tril(torch.ones_like(mask[0], dtype=torch.bool))
        return mask * valid[None]


def distillation_loss(student_logits, teacher_logits, student_maps, teacher_maps,
                      generator=None, points=0, alpha=0.5, beta=0.3, temperature=2.0,
                      budget=0.5, budget_weight=0.02):
    kd = F.kl_div(F.log_softmax(student_logits / temperature, dim=-1),
                  F.softmax(teacher_logits.detach() / temperature, dim=-1),
                  reduction="batchmean") * (temperature ** 2)
    if not points:
        return kd, {"kd": kd.detach().item(), "relation": 0.0, "density": 0.0}
    st = checkpoint_indices(len(student_maps), points)
    tt = checkpoint_indices(len(teacher_maps), points)
    terms, densities = [], []
    for i, (si, ti) in enumerate(zip(st, tt)):
        s = student_maps[si].mean(dim=1)
        t = teacher_maps[ti].detach().mean(dim=1)
        if s.shape != t.shape:
            raise ValueError("Teacher/student token lengths must match")
        m = generator(teacher_maps[ti], student_maps[si], i) if generator is not None else torch.tril(torch.ones_like(s))
        # Mean over valid lower triangle to avoid artificial upper-triangle supervision.
        error = (s - t).square()
        terms.append((error * m).sum() / m.sum().clamp_min(1e-6))
        valid = torch.tril(torch.ones_like(m[0]))
        densities.append((m.sum() / (valid.sum() * m.shape[0])).clamp(0, 1))
    relation = torch.stack(terms).mean()
    density = torch.stack(densities).mean()
    penalty = budget_weight * (density - budget).square() if generator is not None else relation.new_zeros(())
    total = alpha * kd + beta * relation + penalty
    return total, {"kd": kd.detach().item(), "relation": relation.detach().item(), "density": density.detach().item()}
