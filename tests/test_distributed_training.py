"""Global sampling and coordinate normalization contracts for multi-GPU BC."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from lrvla.training_distributed import (
    DistributedEpochBatchSampler, accumulation_for_world, coordinate_denominator, cuda_oom_error, sharded_memory_estimate,
)
from lrvla.training_models import masked_bc_loss, normalize_bc_gradients


@pytest.mark.parametrize("world,accumulation", [(4, 4), (8, 2)])
def test_global_batch_remains_sixteen(world, accumulation):
    assert accumulation_for_world({"batch_size": 1, "gradient_accumulation": 16,
                                   "global_batch_size": 16}, world) == (accumulation, 16)


def test_per_rank_accumulation_without_global_target_is_respected():
    assert accumulation_for_world({"batch_size": 2, "gradient_accumulation": 3}, 4) == (3, 24)
    with pytest.raises(ValueError, match="multiple"):
        accumulation_for_world({"batch_size": 3, "global_batch_size": 16}, 4)


@pytest.mark.parametrize("global_batch,per_gpu", [(64, 16), (32, 8)])
def test_large_physical_batch_uses_no_gradient_accumulation(global_batch, per_gpu):
    assert accumulation_for_world({"batch_size": per_gpu, "gradient_accumulation": 1,
                                   "global_batch_size": global_batch,
                                   "require_no_gradient_accumulation": True}, 4) == (1, global_batch)


def test_no_accumulation_policy_rejects_a_silent_microbatch_substitution():
    with pytest.raises(ValueError, match="forbids gradient accumulation"):
        accumulation_for_world({"batch_size": 8, "global_batch_size": 64,
                                "require_no_gradient_accumulation": True}, 4)


@pytest.mark.parametrize("message,expected", [
    ("CUDA out of memory. Tried to allocate 4 GiB", True),
    ("CUDA error: out of memory", True),
    ("CUDA_ERROR_OUT_OF_MEMORY", True),
    ("DefaultCPUAllocator: can't allocate memory", False),
    ("CUDA error: an illegal memory access was encountered", False),
    ("dataset missing", False),
])
def test_only_cuda_memory_errors_request_a_smaller_batch(message, expected):
    assert cuda_oom_error(RuntimeError(message)) is expected


@pytest.mark.parametrize("world", [4, 8])
def test_each_update_consumes_the_same_global_shuffle_for_four_and_eight_cards(world):
    order = torch.randperm(64, generator=torch.Generator().manual_seed(42)).tolist()
    rank_batches = [list(DistributedEpochBatchSampler(64, 1, 42, rank=rank, world_size=world))
                    for rank in range(world)]
    recovered = [index for microstep in range(len(rank_batches[0])) for rank in range(world)
                 for index in rank_batches[rank][microstep]]
    assert recovered == order
    accumulation = 16 // world
    first_update = [index for microstep in range(accumulation) for rank in range(world)
                    for index in rank_batches[rank][microstep]]
    assert first_update == order[:16]
    assert len(first_update) == len(set(first_update)) == 16


def test_epoch_end_padding_keeps_collectives_balanced_and_resume_exact():
    full = [list(DistributedEpochBatchSampler(11, 2, 7, rank=rank, world_size=4, epoch=2))
            for rank in range(4)]
    assert {len(batches) for batches in full} == {2}
    assert all(len(batch) == 2 for batches in full for batch in batches)
    consumed = [index for step in range(2) for rank in range(4) for index in full[rank][step]]
    assert set(consumed) == set(range(11))
    assert len(consumed) == 16
    for rank in range(4):
        replay = list(DistributedEpochBatchSampler(11, 2, 7, rank=rank, world_size=4, epoch=2, offset=1))
        assert replay == full[rank][1:]


@pytest.mark.parametrize("world", [4, 8])
def test_rank_average_with_unequal_masks_matches_a_single_global_adam_update(world):
    torch.manual_seed(22)
    inputs, target = torch.randn(16, 5, 3), torch.randn(16, 5, 2)
    lengths = torch.arange(16) % 5 + 1
    mask = (torch.arange(5)[None, :, None] < lengths[:, None, None]).expand(-1, -1, 2)
    reference = nn.Linear(3, 2)
    masked_bc_loss(reference(inputs), target, mask).backward()
    ranks = [deepcopy(nn.Linear(3, 2)) for _ in range(world)]
    for model in ranks:
        model.load_state_dict(reference.state_dict())
    for rank, model in enumerate(ranks):
        for index in range(rank, 16, world):
            masked_bc_loss(model(inputs[index:index + 1]), target[index:index + 1],
                           mask[index:index + 1], reduction="sum").backward()
    reduced = deepcopy(reference)
    for index, parameter in enumerate(reduced.parameters()):
        parameter.grad = sum(list(model.parameters())[index].grad for model in ranks) / world
    normalize_bc_gradients(reduced.parameters(), coordinate_denominator(float(mask.sum()), world))
    for actual, expected in zip(reduced.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
    for model in (reference, reduced):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        torch.optim.AdamW(model.parameters(), lr=5e-5, foreach=False).step()
    for actual, expected in zip(reduced.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_memory_accounting_shards_trainables_and_replicates_frozen_weights():
    report = {"trainable_parameters": 2**28, "total_parameters": 3 * 2**28}
    four, eight = sharded_memory_estimate(report, 4), sharded_memory_estimate(report, 8)
    assert four["persistent_gib_per_gpu"] == 2.0
    assert eight["persistent_gib_per_gpu"] == 1.5
    assert four["root_fsdp_temporary_upper_gib_per_gpu"] == 4.0
    assert eight["root_fsdp_temporary_upper_gib_per_gpu"] == 3.5


def test_decoder_layer_sharding_budgets_one_small_flat_group_instead_of_the_whole_expert():
    report = {"trainable_parameters": 2**28, "total_parameters": 3 * 2**28}
    estimate = sharded_memory_estimate(report, 4, largest_shard_parameters=2**20)
    assert estimate["largest_shard_parameters"] == 2**20
    assert estimate["layer_fsdp_temporary_budget_gib_per_gpu"] == 2 + 1 / 64
    assert estimate["layer_fsdp_temporary_budget_gib_per_gpu"] < estimate["root_fsdp_temporary_upper_gib_per_gpu"]
