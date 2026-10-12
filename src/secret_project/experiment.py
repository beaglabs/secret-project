"""Small reproducible experiment. Not a claim of improved real-world KD."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .core import TinyTransformer, MaskGenerator, distillation_loss


def sample(n, length, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab, (n, length), generator=g)
    # A nontrivial sequence rule: compare a distant pair of token values.
    y = (x[:, 1] > x[:, -2]).long()
    return x, y


def run(seed=42, points=7, steps=60, teacher_steps=100, n=384, length=12):
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    x, y = sample(n, length, 32, seed)
    xt, yt = sample(128, length, 32, seed + 40000)
    teacher = TinyTransformer(depth=7, width=48)
    student = TinyTransformer(depth=7, width=24)
    mask = MaskGenerator(points) if points else None
    opt = torch.optim.AdamW(teacher.parameters(), lr=0.003)
    start = time.perf_counter()
    teacher.train()
    for step in range(teacher_steps):
        idx = torch.randint(n, (48,))
        out, _ = teacher(x[idx])
        loss = F.cross_entropy(out, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    params = list(student.parameters()) + (list(mask.parameters()) if mask is not None else [])
    opt = torch.optim.AdamW(params, lr=0.003)
    student.train()
    density = 0.0
    for step in range(steps):
        idx = torch.randint(n, (48,))
        xb, yb = x[idx], y[idx]
        with torch.no_grad():
            t_logits, t_maps = teacher(xb)
        s_logits, s_maps = student(xb)
        kd, stats = distillation_loss(s_logits, t_logits, s_maps, t_maps, mask, points=points)
        loss = F.cross_entropy(s_logits, yb) + kd
        opt.zero_grad(); loss.backward(); opt.step()
        density = stats["density"]
    student.eval()
    with torch.no_grad():
        logits, _ = student(xt)
        teacher_logits, _ = teacher(xt)
        result = {"seed": seed, "points": points, "student_test_accuracy": (logits.argmax(-1) == yt).float().mean().item(),
                  "teacher_test_accuracy": (teacher_logits.argmax(-1) == yt).float().mean().item(),
                  "student_test_ce": F.cross_entropy(logits, yt).item(),
                  "mask_density": density, "elapsed_seconds": round(time.perf_counter() - start, 3),
                  "steps": steps, "teacher_steps": teacher_steps}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=int, choices=[0, 1, 3, 5, 7], default=7)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--teacher-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results/run.json"))
    a = p.parse_args()
    result = run(a.seed, a.points, a.steps, a.teacher_steps)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
