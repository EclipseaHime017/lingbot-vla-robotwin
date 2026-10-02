"""Regression checks for updates with unequal terminal masks and batch sizes."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from lrvla.training import validate
from lrvla.training_checkpoint import resume_training_checkpoint, save_training_checkpoint
from lrvla.training_models import masked_bc_loss, normalize_bc_gradients


@pytest.mark.parametrize("loss_type", ["L1_fm", "fm"])
@pytest.mark.parametrize("sizes", [(8, 8, 8, 8), (8, 3)])
def test_masked_accumulation_matches_one_large_batch_gradient_and_adam_update(loss_type, sizes):
    torch.manual_seed(21)
    samples = sum(sizes)
    inputs = torch.randn(samples, 50, 3)
    target = torch.randn(samples, 50, 2)
    lengths = torch.arange(samples) + 1
    mask = (torch.arange(50)[None, :, None] < lengths[:, None, None]).expand(-1, -1, 2)
    target = target.masked_fill(~mask, 1e6)
    reference = nn.Linear(3, 2)
    accumulated, legacy = deepcopy(reference), deepcopy(reference)
    masked_bc_loss(reference(inputs), target, mask, loss_type).backward()

    start, numerator, count = 0, 0.0, 0
    for size in sizes:
        end = start + size
        chunk_mask = mask[start:end]
        loss_sum = masked_bc_loss(accumulated(inputs[start:end]), target[start:end], chunk_mask,
                                  loss_type, reduction="sum")
        loss_sum.backward()
        numerator += float(loss_sum.detach())
        count += int(chunk_mask.sum())
        (masked_bc_loss(legacy(inputs[start:end]), target[start:end], chunk_mask,
                        loss_type) / len(sizes)).backward()
        start = end
    normalize_bc_gradients(accumulated.parameters(), count)
    expected_loss = float(masked_bc_loss(reference(inputs), target, mask, loss_type).detach())
    assert numerator / count == pytest.approx(expected_loss, rel=1e-6)
    for actual, expected in zip(accumulated.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
    # This deliberately uneven mask reproduces the original weighting bug.
    assert not torch.allclose(legacy.weight.grad, reference.weight.grad)
    for model in (reference, accumulated):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    for actual, expected in zip(accumulated.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("count", [0, -1, float("nan"), float("inf")])
def test_normalization_rejects_invalid_coordinate_count(count):
    with pytest.raises(ValueError, match="valid action count"):
        normalize_bc_gradients(nn.Linear(1, 1).parameters(), count)


def test_validation_uses_coordinate_weighted_mean_and_preserves_random_state():
    model = nn.Linear(1, 1)
    batches = [{"joint_mask": torch.ones(1, 1, 2, dtype=torch.bool)},
               {"joint_mask": torch.ones(1, 4, 2, dtype=torch.bool)}]
    def loss_fn(model, batch):
        torch.randn(3)
        return torch.tensor(2.0 if batch["joint_mask"].shape[1] == 1 else 6.0)
    saved_rng = torch.get_rng_state().clone()
    assert validate(model, batches, "cpu", loss_fn=loss_fn) == pytest.approx(5.2)
    torch.testing.assert_close(torch.get_rng_state(), saved_rng, rtol=0, atol=0)
    assert model.training


@pytest.mark.parametrize("change", ["tail", "loss", "legacy"])
def test_resume_rejects_changed_sampling_or_normalization(tmp_path, change):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    metadata = {"base_identity": {"sha256": "base"}, "manifest_content_sha256": "clean",
                "loss_normalization": "valid_action_coordinate_mean_v1",
                "data_sampling": {"action_chunk_tail": "mask", "batch_size": 8}}
    checkpoint = tmp_path / "training.pt"
    saved = deepcopy(metadata)
    if change == "legacy":
        del saved["data_sampling"]
        del saved["loss_normalization"]
    save_training_checkpoint(model, optimizer, scheduler, checkpoint, {"global_step": 3}, saved)
    expected = deepcopy(metadata)
    if change == "tail":
        expected["data_sampling"]["action_chunk_tail"] = "drop"
    elif change == "loss":
        expected["loss_normalization"] = "other"
    with pytest.raises(ValueError, match="Resume mismatch"):
        resume_training_checkpoint(model, optimizer, scheduler, checkpoint, expected_metadata=expected)
