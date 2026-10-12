import torch
from secret_project.core import TinyTransformer, MaskGenerator, checkpoint_indices, distillation_loss


def test_checkpoints():
    assert checkpoint_indices(28, 7) == [0, 4, 9, 13, 18, 22, 27]
    assert checkpoint_indices(7, 1) == [6]


def test_baseline_and_mask_gradients():
    torch.manual_seed(4)
    teacher = TinyTransformer(depth=7, width=32)
    student = TinyTransformer(depth=7, width=16)
    tokens = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        t_logits, t_maps = teacher(tokens)
    s_logits, s_maps = student(tokens)
    baseline, _ = distillation_loss(s_logits, t_logits, s_maps, t_maps)
    assert torch.isfinite(baseline)
    gen = MaskGenerator(7)
    loss, metrics = distillation_loss(s_logits, t_logits, s_maps, t_maps, gen, points=7)
    assert torch.isfinite(loss)
    assert 0 <= metrics["density"] <= 1
    loss.backward()
    assert gen.net[0].weight.grad is not None
    assert gen.net[0].weight.grad.abs().sum() > 0
    assert student.head.weight.grad is not None


def test_shapes():
    model = TinyTransformer(depth=7, width=16)
    logits, weights = model(torch.randint(0, 32, (2, 6)))
    assert logits.shape == (2, 2)
    assert len(weights) == 7
    assert weights[0].shape == (2, 4, 6, 6)
