"""Two-rank CPU smoke test for variable-length sparse DiFix attention."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mv_unet import UNet2DConditionModel  # noqa: E402
from sparse_attention import build_sparse_attention_context  # noqa: E402


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(11)
    model = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",),
        block_out_channels=(32,),
        layers_per_block=1,
        cross_attention_dim=16,
        attention_head_dim=8,
        norm_num_groups=8,
    )
    model.enable_gradient_checkpointing()
    model.train()
    ddp = DistributedDataParallel(model)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-4)

    pixel_validity = torch.zeros(1, 1, 64, 64)
    if rank == 0:
        pixel_validity[:, :, :, :40] = 1
    context = build_sparse_attention_context(
        pixel_validity,
        num_views=3,
        wrapped_view_index=2,
        latent_shape=(8, 8),
        num_levels=1,
        valid_fraction_threshold=0.8,
        query_modes=("all", "all", "valid_only"),
        key_modes=(
            ("all", "all", "valid_only"),
            ("all", "all", "valid_only"),
            ("all", "all", "valid_only"),
        ),
    )
    ddp.module.set_multiview_attention_context(context)
    sample = torch.randn(3, 4, 8, 8)
    text = torch.randn(3, 5, 16)
    output = ddp(sample, torch.tensor(199), text).sample
    loss = output.square().mean()
    loss.backward()
    optimizer.step()

    flat = torch.cat(
        [parameter.detach().reshape(-1) for parameter in ddp.parameters()]
    )
    gathered = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flat)
    max_difference = max(
        float((gathered[0] - other).abs().max()) for other in gathered[1:]
    )
    if not torch.isfinite(output).all():
        raise RuntimeError(f"rank {rank}: non-finite sparse output")
    if max_difference != 0.0:
        raise RuntimeError(
            f"DDP parameters diverged; max difference={max_difference}"
        )
    if rank == 0:
        print(
            "sparse DiFix DDP smoke passed: unequal valid-token counts, "
            f"max parameter difference={max_difference}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" not in os.environ:
        raise SystemExit(
            "Run with: torchrun --standalone --nproc_per_node=2 "
            "scripts/smoke_sparse_attention_ddp.py"
        )
    main()
