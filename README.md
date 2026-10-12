# Secret Project — Attention-Mask Distillation Lab

Reproducible research scaffold for **1/3/5/7-point learned attention-mask distillation**. The mask generator trains only during distillation; student inference does not depend on it.

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m secret_project.experiment --steps 60 --points 7 --output results/seven.json
python -m secret_project.experiment --steps 60 --points 0 --output results/baseline.json
```

Runs on CPU using a deterministic synthetic sequence classification task and small transformer teacher/student. Teacher is pretrained *locally on the synthetic task* before being frozen; it is not a randomly initialized oracle.

## Gemma 4 E4B teacher

```bash
pip install -e ".[gemma]"
python -m secret_project.gemma --model google/gemma-4-E4B-it --prompt "Which is larger: 3 or 7?" --output results/gemma-attention.json
```

This **opt-in** command downloads the checkpoint when allowed, and records seven selected layer indices, attention summary statistics and shapes, not enormous tensors. Hugging Face access and substantial RAM/VRAM are required. Gemma's local/global attention pattern is *not* treated as identical layers. For actual Gemma-supervised distillation, data must be token-aligned and teacher attentions/logits must be cached or streamed into a matching student pipeline. The included toy experiment is **not** a Gemma-trained student.

## Research design

- Modes: points=0 (logit KD baseline), 1, 3, 5, 7; shared learned mask generator and seven checkpoints.
- Attention supervision uses teacher/student attention maps at evenly spaced layers, averaging over heads to handle different head counts.
- Mask is a **training loss selector**, not an inference-time intervention. Masks are normalized over valid causal edges; sparsity is controlled via a budget penalty. Mask-generator gradients flow from student-relative masked attention mismatch. The generator can otherwise exploit the objective by emphasizing easy edges; evaluate against uniform attention transfer as a control.
- Metrics: test accuracy, cross entropy, student training loss, mask density, wall-clock time. For research claims run matched seeds (>=3), matched teacher/data, and equal compute budgets. Synthetic benchmark is only plumbing, not evidence of useful distillation.
- Long sequences create quadratic attention memory; fused attention may not return attention maps. Begin with short sequences, explicit eager attention, and never infer performance from a single seed.
- No model weights committed. GitHub CI runs only small smoke tests.

## Structure

```
src/secret_project/{core,experiment,gemma}.py
tests/test_core.py
.github/workflows/ci.yml
```

## Scope

This is an experimental implementation, not validated performance improvement. In particular, the demo mask is student-aware through loss gradients, but it is **not** a full bilevel meta-learning algorithm. A later controlled study should compare learned masking with uniform relation transfer, optimize mask updates on held-out validation batches, and use Gemma outputs directly.
