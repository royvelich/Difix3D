import argparse
import gc
import json
import math
import os
import random
from glob import glob

import diffusers
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.transforms.functional import crop

from dataset import PairedDataset
from loss import gram_loss
from metrics import batch_metrics, iqa_backend_metadata
from model import Difix, load_ckpt_from_state_dict, save_ckpt
from visualization import build_labeled_grid, save_grid, to_uint8_panel
from wrapped_patch_oracle import override_render_with_wrapped_patches


TWO_VIEW_PANEL_TITLES = (
    "corrupt",
    "reference",
    "clean",
    "restored render",
    "restored reference",
)

THREE_VIEW_PANEL_TITLES = (
    "corrupt",
    "reference",
    "wrapped reference",
    "wrapped validity",
    "clean",
    "restored render",
    "restored reference",
    "restored wrapped",
)

ORACLE_THREE_VIEW_PANEL_TITLES = (
    "corrupt",
    "reference",
    "wrapped reference",
    "wrapped validity",
    "wrapped + oracle selection",
    "oracle-combined render",
    "clean",
    "restored render",
    "restored reference",
    "restored wrapped",
)


def _interval_due(step, frequency):
    return frequency > 0 and step % frequency == 0


def _checkpoint_step(path):
    filename = os.path.basename(path)
    return int(filename.replace("model_", "").replace(".pkl", ""))


def _find_resume_checkpoint(resume):
    if os.path.isdir(resume):
        checkpoints = glob(os.path.join(resume, "model_*.pkl"))
        if not checkpoints:
            raise FileNotFoundError(f"No model_*.pkl checkpoints found in {resume}")
        return max(checkpoints, key=_checkpoint_step)
    if resume.endswith(".pkl") and os.path.isfile(resume):
        return resume
    raise ValueError(f"Invalid resume checkpoint or directory: {resume}")


class _StridedValidationSampler(torch.utils.data.Sampler):
    """Partition validation indices across ranks without padding duplicates."""

    def __init__(self, dataset, process_index, num_processes):
        if num_processes < 1:
            raise ValueError("num_processes must be >= 1")
        if process_index < 0 or process_index >= num_processes:
            raise ValueError(
                "process_index must be in [0, num_processes)"
            )
        self.dataset = dataset
        self.process_index = int(process_index)
        self.num_processes = int(num_processes)

    def __iter__(self):
        return iter(
            range(
                self.process_index,
                len(self.dataset),
                self.num_processes,
            )
        )

    def __len__(self):
        remaining = len(self.dataset) - self.process_index
        if remaining <= 0:
            return 0
        return (remaining + self.num_processes - 1) // self.num_processes


def _image_panel(
    conditioning,
    target,
    prediction,
    index,
    caption,
    wrapped_reference_validity=None,
    original_conditioning=None,
    render_override_mask=None,
):
    if conditioning.shape[1] == 2:
        panel_titles = TWO_VIEW_PANEL_TITLES
        panels = [
            to_uint8_panel(conditioning[index, 0]),
            to_uint8_panel(conditioning[index, 1]),
            to_uint8_panel(target[index, 0]),
            to_uint8_panel(prediction[index, 0]),
            to_uint8_panel(prediction[index, 1]),
        ]
    elif conditioning.shape[1] == 3:
        if wrapped_reference_validity is None:
            raise ValueError(
                "Three-view panels require wrapped_reference_validity"
            )
        validity_panel = (
            wrapped_reference_validity[index].repeat(3, 1, 1) * 2.0 - 1.0
        )
        if render_override_mask is None:
            panel_titles = THREE_VIEW_PANEL_TITLES
            panels = [
                to_uint8_panel(conditioning[index, 0]),
                to_uint8_panel(conditioning[index, 1]),
                to_uint8_panel(conditioning[index, 2]),
                to_uint8_panel(validity_panel),
                to_uint8_panel(target[index, 0]),
                to_uint8_panel(prediction[index, 0]),
                to_uint8_panel(prediction[index, 1]),
                to_uint8_panel(prediction[index, 2]),
            ]
        else:
            if original_conditioning is None:
                raise ValueError(
                    "Oracle panels require original_conditioning"
                )
            selected = render_override_mask[index].bool()
            red = conditioning.new_tensor([1.0, -1.0, -1.0]).view(3, 1, 1)
            wrapped_overlay = torch.where(
                selected,
                conditioning[index, 2] * 0.55 + red * 0.45,
                conditioning[index, 2],
            )
            panel_titles = ORACLE_THREE_VIEW_PANEL_TITLES
            panels = [
                to_uint8_panel(original_conditioning[index, 0]),
                to_uint8_panel(conditioning[index, 1]),
                to_uint8_panel(conditioning[index, 2]),
                to_uint8_panel(validity_panel),
                to_uint8_panel(wrapped_overlay),
                to_uint8_panel(conditioning[index, 0]),
                to_uint8_panel(target[index, 0]),
                to_uint8_panel(prediction[index, 0]),
                to_uint8_panel(prediction[index, 1]),
                to_uint8_panel(prediction[index, 2]),
            ]
    else:
        raise ValueError(
            "DiFix panels support two or three views, got "
            f"{conditioning.shape[1]}"
        )

    row = {
        "panels": panels,
        "caption": caption,
    }
    return build_labeled_grid(
        [row],
        panel_titles,
    )


def _panel_titles(num_views, oracle_override=False):
    if num_views == 2:
        return TWO_VIEW_PANEL_TITLES
    if num_views == 3:
        if oracle_override:
            return ORACLE_THREE_VIEW_PANEL_TITLES
        return THREE_VIEW_PANEL_TITLES
    raise ValueError(f"Unsupported DiFix view count {num_views}")


def _prepare_model_conditioning(
    conditioning,
    target,
    wrapped_reference_validity,
    args,
):
    """Build the optional oracle-combined render using shared train/val code."""

    if args.render_endpoint_mode == "original":
        return conditioning, None
    if args.render_endpoint_mode != "wrapped_patch_override":
        raise ValueError(
            "render_endpoint_mode must be original|wrapped_patch_override"
        )
    if conditioning.shape[1] != 3:
        raise ValueError(
            "wrapped-patch render replacement requires three conditioning "
            f"views, got {conditioning.shape[1]}"
        )
    if wrapped_reference_validity is None:
        raise ValueError(
            "wrapped-patch render replacement requires wrapped validity"
        )

    combined, selected_patches, eligible_patches = (
        override_render_with_wrapped_patches(
            conditioning[:, 0],
            conditioning[:, 2],
            wrapped_reference_validity,
            patch_size=args.render_endpoint_override_patch_size,
            min_valid_fraction=args.render_endpoint_override_valid_fraction,
            selection_policy=args.render_endpoint_override_policy,
            clean=target[:, 0],
            oracle_metric=args.render_endpoint_oracle_metric,
            oracle_window_patches=(
                args.render_endpoint_oracle_window_patches
            ),
            oracle_margin=args.render_endpoint_oracle_margin,
            oracle_relative_gain=(
                args.render_endpoint_oracle_relative_gain
            ),
            oracle_lpips_chunk_size=(
                args.render_endpoint_oracle_lpips_chunk_size
            ),
        )
    )
    model_conditioning = conditioning.clone()
    model_conditioning[:, 0] = combined

    patch_area = float(args.render_endpoint_override_patch_size**2)
    total_patch_count = selected_patches.numel() / patch_area
    selected_patch_count = selected_patches.float().sum() / patch_area
    eligible_patch_count = eligible_patches.float().sum() / patch_area
    selected_fraction = selected_patch_count / total_patch_count
    eligible_fraction = eligible_patch_count / total_patch_count
    selected_of_eligible = (
        selected_patch_count / eligible_patch_count.clamp_min(1.0)
    )
    return model_conditioning, {
        "selected_patches": selected_patches,
        "eligible_patches": eligible_patches,
        "copy_mask": (
            selected_patches
            & wrapped_reference_validity.to(selected_patches.device).gt(0)
        ),
        "selected_patch_count": selected_patch_count,
        "eligible_patch_count": eligible_patch_count,
        "selected_fraction": selected_fraction,
        "eligible_fraction": eligible_fraction,
        "selected_of_eligible": selected_of_eligible,
    }


def _log_image(tracker, tag, image, step):
    if tracker.name == "tensorboard":
        tracker.writer.add_image(tag, image, step, dataformats="HWC")
    elif tracker.name == "wandb":
        import wandb

        tracker.log({tag: wandb.Image(image)}, step=step)
    elif tracker.name == "clearml":
        tracker.log_images({tag: image}, step=step)


def _log_metric_scalars(
    tracker,
    average_metrics,
    median_metrics,
    step,
    namespace="restore-val",
):
    values = {
        f"{namespace}/metrics/{name}": value
        for name, value in average_metrics.items()
    }
    values.update(
        {
            f"{namespace}/metrics/{name}_median": value
            for name, value in median_metrics.items()
        }
    )
    if tracker.name == "tensorboard":
        for name, value in values.items():
            tracker.writer.add_scalar(name, value, step)
    elif tracker.name in ("wandb", "clearml"):
        tracker.log(values, step=step)


def _aggregate_metric_rows(rows):
    average_metrics = {}
    median_metrics = {}
    metric_counts = {}
    metric_names = {
        name for row in rows for name in row.get("metrics", {}).keys()
    }
    for name in sorted(metric_names):
        values = [
            float(row["metrics"][name])
            for row in rows
            if name in row.get("metrics", {})
        ]
        if values:
            array = np.asarray(values, dtype=np.float64)
            average_metrics[name] = float(np.mean(array))
            median_metrics[name] = float(np.median(array))
            metric_counts[name] = len(values)
    return average_metrics, median_metrics, metric_counts


def _gather_rows(accelerator, local_rows):
    if accelerator.num_processes == 1:
        return local_rows
    gathered = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(gathered, local_rows)
    return [row for process_rows in gathered for row in process_rows]


def _gather_mean(accelerator, value):
    if not torch.is_tensor(value):
        value = torch.tensor(value, device=accelerator.device)
    gathered = accelerator.gather(value.detach().float().reshape(1))
    return float(gathered.mean().item())


def _gather_ratio(accelerator, numerator, denominator):
    numerator = accelerator.gather(
        numerator.detach().float().reshape(1)
    ).sum()
    denominator = accelerator.gather(
        denominator.detach().float().reshape(1)
    ).sum()
    if denominator.item() <= 0:
        return 0.0
    return float((numerator / denominator).item())


def _wrapped_loss_tensors(
    prediction,
    target,
    wrapped_reference_validity,
    mask_mode,
):
    """Return L2 and perceptual inputs with independent wrapped supervision."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target mismatch: {prediction.shape} vs {target.shape}"
        )
    if prediction.shape[1] != 3:
        prediction_flat = rearrange(
            prediction, "b v c h w -> (b v) c h w"
        )
        target_flat = rearrange(
            target, "b v c h w -> (b v) c h w"
        )
        return (
            F.mse_loss(
                prediction_flat.float(),
                target_flat.float(),
                reduction="mean",
            ),
            prediction_flat,
            target_flat,
            torch.ones(
                prediction_flat.shape[0],
                device=prediction.device,
                dtype=prediction.dtype,
            ),
        )
    if wrapped_reference_validity is None:
        raise ValueError(
            "Three-view loss requires wrapped_reference_validity"
        )
    if mask_mode not in ("all", "valid_pixels"):
        raise ValueError(
            "wrapped_reference_loss_mask must be all|valid_pixels"
        )

    view_l2 = [
        F.mse_loss(
            prediction[:, view].float(),
            target[:, view].float(),
            reduction="mean",
        )
        for view in (0, 1)
    ]
    perceptual_prediction = prediction
    wrapped_row_weight = torch.ones(
        prediction.shape[0],
        device=prediction.device,
        dtype=prediction.dtype,
    )
    if mask_mode == "valid_pixels":
        validity = wrapped_reference_validity.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        error = (
            prediction[:, 2].float() - target[:, 2].float()
        ).square()
        denominator = (
            validity.float().sum() * prediction.shape[2]
        ).clamp_min(1.0)
        view_l2.append(
            (error * validity.float()).sum() / denominator
        )
        wrapped_row_weight = (
            validity.flatten(1).any(dim=1).to(prediction.dtype)
        )
        wrapped_for_perceptual = (
            prediction[:, 2] * validity
            + target[:, 2] * (1.0 - validity)
        )
        perceptual_prediction = torch.cat(
            [
                prediction[:, :2],
                wrapped_for_perceptual.unsqueeze(1),
            ],
            dim=1,
        )
    else:
        view_l2.append(
            F.mse_loss(
                prediction[:, 2].float(),
                target[:, 2].float(),
                reduction="mean",
            )
        )

    view_weights = torch.stack(
        [
            torch.ones((), device=prediction.device),
            torch.ones((), device=prediction.device),
            wrapped_row_weight.any().float(),
        ]
    )
    weighted_l2 = (
        torch.stack(view_l2) * view_weights
    ).sum() / view_weights.sum().clamp_min(1.0)

    return (
        weighted_l2,
        rearrange(
            perceptual_prediction, "b v c h w -> (b v) c h w"
        ),
        rearrange(target, "b v c h w -> (b v) c h w"),
        torch.stack(
            [
                torch.ones_like(wrapped_row_weight),
                torch.ones_like(wrapped_row_weight),
                wrapped_row_weight,
            ],
            dim=1,
        ).reshape(-1),
    )


def _valid_pixel_metrics(prediction, target, validity, prefix):
    """Score wrapped output while excluding unsupervised warp holes."""

    expected = (prediction.shape[0], 1, *prediction.shape[-2:])
    if tuple(validity.shape) != expected:
        raise ValueError(
            f"validity must be {expected}, got {tuple(validity.shape)}"
        )
    valid = validity.to(device=prediction.device).gt(0)
    metrics = {
        f"{prefix}_coverage": float(valid.float().mean().item()),
        f"{prefix}_active": float(valid.any().item()),
    }
    if not valid.any():
        return metrics

    # Perceptual backends have no reliable masked spatial reduction. Fill
    # excluded pixels from the target, log coverage, and override PSNR with
    # the intuitive value normalized only over valid RGB pixels.
    target_filled = torch.where(valid, prediction, target)
    metrics.update(
        {
            f"{prefix}_{name}": value
            for name, value in batch_metrics(target_filled, target).items()
        }
    )
    selected_error = (
        (prediction.float() - target.float()) * 0.5
    )[valid.expand_as(prediction)]
    mse = selected_error.square().mean()
    metrics.update(
        {
            f"{prefix}_mae": float(selected_error.abs().mean().item()),
            f"{prefix}_rmse": float(mse.sqrt().item()),
            f"{prefix}_psnr": float(
                (-10.0 * torch.log10(mse.clamp_min(1e-12))).item()
            ),
        }
    )
    return metrics


def _dataset_kwargs(
    args,
    data_root=None,
    wrapped_reference_cache_dir=None,
    wrapped_reference_depth_cache_dir=None,
):
    return {
        "height": args.height,
        "width": args.width,
        "prompt_override": args.prompt,
        "default_prompt": args.default_prompt,
        "require_reference": True,
        "data_root": args.data_root if data_root is None else data_root,
        "path_prefix_from": args.path_prefix_from,
        "path_prefix_to": args.path_prefix_to,
        "wrapped_reference_cache_dir": wrapped_reference_cache_dir,
        "wrapped_reference_fill": args.wrapped_reference_fill,
        "wrapped_reference_builder_repo": (
            args.wrapped_reference_builder_repo
        ),
        "wrapped_reference_depth_cache_dir": (
            args.wrapped_reference_depth_cache_dir
            if wrapped_reference_depth_cache_dir is None
            else wrapped_reference_depth_cache_dir
        ),
        "wrapped_reference_splat_radius": (
            args.wrapped_reference_splat_radius
        ),
        "cache_built_wrapped_references": (
            args.cache_built_wrapped_references
        ),
        "storage_backend": args.storage_backend,
        "s3_cache_dir": args.s3_cache_dir,
        "s3_cache_images": args.s3_cache_images,
        "s3_max_pool_connections": args.s3_max_pool_connections,
        "s3_tar_routes_path": args.s3_tar_routes_path,
    }


def _build_validation_loader(args, tokenizer, process_index=0, num_processes=1):
    validation_path = args.validation_dataset_path or args.dataset_path
    dataset = PairedDataset(
        dataset_path=validation_path,
        split=args.validation_split,
        tokenizer=tokenizer,
        **_dataset_kwargs(
            args,
            data_root=args.validation_data_root or args.data_root,
            wrapped_reference_cache_dir=(
                args.validation_wrapped_reference_cache_dir
                or args.wrapped_reference_cache_dir
            ),
            wrapped_reference_depth_cache_dir=(
                args.validation_wrapped_reference_depth_cache_dir
                or args.wrapped_reference_depth_cache_dir
            ),
        ),
    )
    if len(dataset) == 0:
        raise ValueError(
            f"Validation split {args.validation_split!r} in "
            f"{validation_path} is empty"
        )
    sampler = _StridedValidationSampler(
        dataset,
        process_index=process_index,
        num_processes=num_processes,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.validation_num_workers,
        pin_memory=True,
    )


def _run_validation(
    accelerator,
    net_difix,
    dataloader,
    weight_dtype,
    num_samples,
    num_panel_samples,
    output_dir,
    step,
    args,
):
    model = accelerator.unwrap_model(net_difix)
    model.eval()
    local_rows = []
    row_dir = os.path.join(output_dir, "eval", f"step_{step:09d}")
    if accelerator.is_main_process:
        os.makedirs(row_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    for local_index, batch in enumerate(dataloader):
        sample_index = (
            accelerator.process_index
            + local_index * accelerator.num_processes
        )
        if sample_index >= num_samples:
            break

        # Keep source pixels in fp32 for the GT oracle and validation metrics.
        # Autocast handles the model call without rounding the decision inputs.
        conditioning = batch["conditioning_pixel_values"].to(
            accelerator.device
        )
        target = batch["output_pixel_values"].to(accelerator.device)
        input_ids = batch["input_ids"].to(accelerator.device)
        wrapped_reference_validity = batch.get(
            "wrapped_reference_validity"
        )
        if wrapped_reference_validity is not None:
            wrapped_reference_validity = wrapped_reference_validity.to(
                accelerator.device,
            )

        original_conditioning = conditioning
        conditioning, override_info = _prepare_model_conditioning(
            conditioning,
            target,
            wrapped_reference_validity,
            args,
        )

        with torch.no_grad(), accelerator.autocast():
            prediction = model(
                conditioning,
                prompt_tokens=input_ids,
                wrapped_reference_validity=wrapped_reference_validity,
            )
        render_metrics = batch_metrics(prediction[:, 0], target[:, 0])
        input_metrics = batch_metrics(
            original_conditioning[:, 0],
            target[:, 0],
        )
        reference_metrics = batch_metrics(
            prediction[:, 1],
            target[:, 1],
        )
        metrics = {
            f"ours_ref_{name}": value
            for name, value in render_metrics.items()
        }
        metrics.update(
            {
                f"input_{name}": value
                for name, value in input_metrics.items()
            }
        )
        metrics.update(
            {
                f"restored_reference_{name}": value
                for name, value in reference_metrics.items()
            }
        )
        if prediction.shape[1] == 3:
            wrapped_metrics = batch_metrics(
                prediction[:, 2],
                target[:, 2],
            )
            metrics.update(
                {
                    f"restored_wrapped_{name}": value
                    for name, value in wrapped_metrics.items()
                }
            )
            metrics.update(
                _valid_pixel_metrics(
                    prediction[:, 2],
                    target[:, 2],
                    wrapped_reference_validity,
                    "restored_wrapped_valid",
                )
            )
            metrics["wrapped_valid_fraction"] = float(
                wrapped_reference_validity.float().mean().item()
            )
        if override_info is not None:
            combined_metrics = batch_metrics(
                conditioning[:, 0],
                target[:, 0],
            )
            metrics.update(
                {
                    f"oracle_combined_{name}": value
                    for name, value in combined_metrics.items()
                }
            )
            metrics.update(
                {
                    "oracle_override_patch_fraction": float(
                        override_info["selected_fraction"].item()
                    ),
                    "oracle_eligible_patch_fraction": float(
                        override_info["eligible_fraction"].item()
                    ),
                }
            )
            if override_info["eligible_patch_count"].item() > 0:
                metrics["oracle_selected_of_eligible"] = float(
                    override_info["selected_of_eligible"].item()
                )

        record_id = batch["record_id"][0]
        row_path = None
        if sample_index < num_panel_samples:
            panel = _image_panel(
                conditioning,
                target,
                prediction,
                0,
                str(record_id),
                wrapped_reference_validity,
                original_conditioning=original_conditioning,
                render_override_mask=(
                    override_info["copy_mask"]
                    if override_info is not None
                    else None
                ),
            )
            row_path = save_grid(
                panel,
                os.path.join(row_dir, f"pair_{sample_index:03d}.webp"),
            )

        local_rows.append(
            {
                "index": sample_index,
                "record_id": str(record_id),
                "row_path": row_path,
                "metrics": metrics,
                "oracle_selected_patch_count": (
                    float(override_info["selected_patch_count"].item())
                    if override_info is not None
                    else None
                ),
                "oracle_eligible_patch_count": (
                    float(override_info["eligible_patch_count"].item())
                    if override_info is not None
                    else None
                ),
            }
        )

    rows = sorted(
        _gather_rows(accelerator, local_rows),
        key=lambda row: row["index"],
    )
    average_metrics, median_metrics, metric_counts = _aggregate_metric_rows(
        rows
    )
    oracle_rows = [
        row
        for row in rows
        if row.get("oracle_eligible_patch_count") is not None
    ]
    if oracle_rows:
        selected_total = sum(
            row["oracle_selected_patch_count"] for row in oracle_rows
        )
        eligible_total = sum(
            row["oracle_eligible_patch_count"] for row in oracle_rows
        )
        average_metrics["oracle_selected_of_eligible"] = (
            selected_total / eligible_total if eligible_total > 0 else 0.0
        )
        metric_counts["oracle_selected_of_eligible"] = sum(
            row["oracle_eligible_patch_count"] > 0 for row in oracle_rows
        )

    if not rows:
        raise ValueError("Validation produced no samples")

    if accelerator.is_main_process:
        panel_titles = _panel_titles(
            3
            if any("restored_wrapped_psnr" in row["metrics"] for row in rows)
            else 2,
            oracle_override=any(
                "oracle_override_patch_fraction" in row["metrics"]
                for row in rows
            ),
        )
        summary_path = os.path.join(row_dir, "validation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "namespace": "restore-val",
                    "step": step,
                    "num_pairs": len(rows),
                    "columns": list(panel_titles),
                    "iqa_backend": iqa_backend_metadata(),
                    "average_metrics": average_metrics,
                    "median_metrics": median_metrics,
                    "metric_counts": metric_counts,
                    "pairs": rows,
                },
                handle,
                indent=2,
                sort_keys=True,
            )

        trackers = accelerator.trackers if accelerator.trackers else []
        image_tag = "restore-val/" + " | ".join(panel_titles)
        for row in rows:
            if row["row_path"] is None:
                continue
            from PIL import Image

            with Image.open(row["row_path"]) as saved:
                image = np.asarray(saved.convert("RGB"), dtype=np.uint8)
            for tracker in trackers:
                try:
                    _log_image(
                        tracker,
                        f"{image_tag}/pair_{row['index']:02d}",
                        image,
                        step,
                    )
                except Exception as exc:
                    print(
                        "[restore-val] tracker image upload failed "
                        f"({tracker.name}, pair {row['index']}): {exc}"
                    )

        for tracker in trackers:
            try:
                _log_metric_scalars(
                    tracker,
                    average_metrics,
                    median_metrics,
                    step,
                )
            except Exception as exc:
                print(
                    "[restore-val] tracker metric logging failed "
                    f"({tracker.name}): {exc}"
                )

        summary_bits = []
        for prefix in (
            "input",
            "ours_ref",
            "restored_reference",
            "restored_wrapped",
            "restored_wrapped_valid",
            "oracle_combined",
        ):
            if f"{prefix}_psnr" not in average_metrics:
                continue
            summary_bits.append(
                f"{prefix}: "
                f"PSNR={average_metrics[f'{prefix}_psnr']:.2f} "
                f"SSIM={average_metrics[f'{prefix}_ssim']:.3f} "
                f"LPIPS={average_metrics[f'{prefix}_lpips']:.3f}"
            )
        print(
            f"[restore-val] step {step}: "
            + "  ".join(summary_bits)
            + f"  panels={row_dir}"
        )

    accelerator.wait_for_everyone()
    model.set_train()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args):
    if not args.mv_unet:
        raise ValueError(
            "Vanilla reference-conditioned Difix requires --mv_unet. "
            "Use the default setting or pass --mv_unet explicitly."
        )
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("--height and --width must both be divisible by 8")
    if (args.path_prefix_from is None) != (args.path_prefix_to is None):
        raise ValueError(
            "--path_prefix_from and --path_prefix_to must be provided together"
        )
    if args.storage_backend == "s3_stream" and not args.s3_cache_dir:
        raise ValueError(
            "--storage_backend=s3_stream requires --s3_cache_dir"
        )
    if args.s3_tar_routes_path and args.storage_backend != "s3_stream":
        raise ValueError(
            "--s3_tar_routes_path requires --storage_backend=s3_stream"
        )
    wrapped_mode = args.wrapped_reference_cache_dir is not None
    if args.sparse_mv_attention and not wrapped_mode:
        raise ValueError(
            "--sparse_mv_attention requires --wrapped_reference_cache_dir"
        )
    if (
        args.wrapped_reference_builder_repo is None
    ) != (
        args.wrapped_reference_depth_cache_dir is None
    ):
        raise ValueError(
            "--wrapped_reference_builder_repo and "
            "--wrapped_reference_depth_cache_dir must be provided together"
        )
    if (
        args.validation_wrapped_reference_cache_dir is not None
        and args.eval_freq <= 0
    ):
        raise ValueError(
            "--validation_wrapped_reference_cache_dir requires validation"
        )
    if (
        args.validation_wrapped_reference_cache_dir is not None
        and not wrapped_mode
    ):
        raise ValueError(
            "--validation_wrapped_reference_cache_dir requires the training "
            "--wrapped_reference_cache_dir so train and validation use the "
            "same number of views"
        )
    if (
        args.wrapped_reference_denoise_mask == "valid_only"
        and not args.sparse_mv_attention
    ):
        raise ValueError(
            "valid-only wrapped denoising requires --sparse_mv_attention"
        )
    if args.render_endpoint_mode == "wrapped_patch_override":
        if not wrapped_mode:
            raise ValueError(
                "--render_endpoint_mode=wrapped_patch_override requires "
                "--wrapped_reference_cache_dir"
            )
        if (
            args.height % args.render_endpoint_override_patch_size
            or args.width % args.render_endpoint_override_patch_size
        ):
            raise ValueError(
                "--render_endpoint_override_patch_size must divide both "
                "--height and --width"
            )
    elif args.render_endpoint_override_policy != "all_valid":
        raise ValueError(
            "--render_endpoint_override_policy requires "
            "--render_endpoint_mode=wrapped_patch_override"
        )
    if args.render_endpoint_override_patch_size < 1:
        raise ValueError(
            "--render_endpoint_override_patch_size must be >= 1"
        )
    if not 0.0 <= args.render_endpoint_override_valid_fraction <= 1.0:
        raise ValueError(
            "--render_endpoint_override_valid_fraction must be in [0, 1]"
        )
    if (
        args.render_endpoint_oracle_window_patches < 1
        or args.render_endpoint_oracle_window_patches % 2 != 1
    ):
        raise ValueError(
            "--render_endpoint_oracle_window_patches must be a positive "
            "odd integer"
        )
    if (
        not math.isfinite(args.render_endpoint_oracle_margin)
        or args.render_endpoint_oracle_margin < 0.0
    ):
        raise ValueError(
            "--render_endpoint_oracle_margin must be finite and >= 0"
        )
    if not (
        math.isfinite(args.render_endpoint_oracle_relative_gain)
        and 0.0 <= args.render_endpoint_oracle_relative_gain < 1.0
    ):
        raise ValueError(
            "--render_endpoint_oracle_relative_gain must be finite and "
            "in [0, 1)"
        )
    if args.render_endpoint_oracle_lpips_chunk_size < 1:
        raise ValueError(
            "--render_endpoint_oracle_lpips_chunk_size must be >= 1"
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    accelerator.wait_for_everyone()

    net_difix = Difix(
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        mv_unet=True,
        base_model_name_or_path=args.pretrained_model_name_or_path,
        sparse_mv_attention=args.sparse_mv_attention,
        view_attention_query_modes=args.view_attention_query_modes,
        view_attention_matrix=args.view_attention_matrix,
        wrapped_reference_patch_valid_fraction=(
            args.wrapped_reference_patch_valid_fraction
        ),
        wrapped_reference_denoise_mask=(
            args.wrapped_reference_denoise_mask
        ),
    )
    net_difix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError(
                "xformers is not available; install it or remove "
                "--enable_xformers_memory_efficient_attention"
            )
        net_difix.unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        net_difix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_train = PairedDataset(
        dataset_path=args.dataset_path,
        split=args.train_split,
        tokenizer=net_difix.tokenizer,
        **_dataset_kwargs(
            args,
            wrapped_reference_cache_dir=args.wrapped_reference_cache_dir,
        ),
    )
    if len(dataset_train) == 0:
        raise ValueError(
            f"Training split {args.train_split!r} in {args.dataset_path} is empty"
        )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    dataloader_val = (
        _build_validation_loader(
            args,
            net_difix.tokenizer,
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
        )
        if args.eval_freq > 0
        else None
    )

    net_lpips = lpips.LPIPS(net="vgg")
    net_lpips.requires_grad_(False)
    net_lpips.eval()

    net_vgg = torchvision.models.vgg16(pretrained=True).features
    net_vgg.requires_grad_(False)
    net_vgg.eval()

    layers_to_optimize = [
        parameter for parameter in net_difix.parameters() if parameter.requires_grad
    ]
    if not layers_to_optimize:
        raise RuntimeError("The model has no trainable parameters")

    optimizer = torch.optim.AdamW(
        layers_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    global_step = 0
    if args.resume is not None:
        checkpoint_path = _find_resume_checkpoint(args.resume)
        global_step = _checkpoint_step(checkpoint_path)
        if accelerator.is_main_process:
            print(f"Resuming from {checkpoint_path} at step {global_step}")
        net_difix, optimizer = load_ckpt_from_state_dict(
            net_difix,
            optimizer,
            checkpoint_path,
            lr_scheduler=lr_scheduler,
        )
    elif accelerator.is_main_process:
        print(
            "Starting from the SD-Turbo base weights; no Difix checkpoint "
            "is being resumed"
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_difix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device)
    net_vgg_on_device = False
    if (
        args.lambda_gram > 0
        and global_step > args.gram_loss_warmup_steps
    ):
        net_vgg.to(accelerator.device)
        net_vgg_on_device = True

    net_difix, optimizer, dataloader_train, lr_scheduler = accelerator.prepare(
        net_difix, optimizer, dataloader_train, lr_scheduler
    )
    imagenet_normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )

    if accelerator.is_main_process and args.report_to != "none":
        init_kwargs = {}
        if args.report_to in ("wandb", "all"):
            init_kwargs["wandb"] = {
                "name": args.tracker_run_name,
                "dir": args.output_dir,
            }
        if args.report_to in ("clearml", "all"):
            init_kwargs["clearml"] = {
                "project_name": args.tracker_project_name,
                "task_name": args.tracker_run_name,
            }
        accelerator.init_trackers(
            args.tracker_project_name,
            config=dict(vars(args)),
            init_kwargs=init_kwargs,
        )

    if accelerator.is_main_process:
        effective_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        print(
            f"train records={len(dataset_train)}, "
            f"effective global batch={effective_batch_size}, "
            f"target optimizer steps={args.max_train_steps}"
        )

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=min(global_step, args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    if args.validate_on_start:
        if dataloader_val is None:
            raise ValueError(
                "--validate_on_start requires validation to be enabled "
                "with --eval_freq > 0"
            )
        accelerator.wait_for_everyone()
        _run_validation(
            accelerator,
            net_difix,
            dataloader_val,
            weight_dtype,
            args.num_samples_eval,
            args.num_validation_panel_samples,
            args.output_dir,
            global_step,
            args,
        )
        accelerator.wait_for_everyone()

    updates_per_epoch = math.ceil(
        len(dataloader_train) / args.gradient_accumulation_steps
    )
    epochs_required = math.ceil(
        max(args.max_train_steps - global_step, 0) / max(updates_per_epoch, 1)
    )

    for epoch in range(epochs_required):
        for batch in dataloader_train:
            grad_norm = None
            with accelerator.accumulate(net_difix):
                conditioning = batch["conditioning_pixel_values"]
                target = batch["output_pixel_values"]
                wrapped_reference_validity = batch.get(
                    "wrapped_reference_validity"
                )
                batch_size, num_views, _, height, width = conditioning.shape
                original_conditioning = conditioning
                conditioning, override_info = _prepare_model_conditioning(
                    conditioning,
                    target,
                    wrapped_reference_validity,
                    args,
                )

                with accelerator.autocast():
                    prediction = net_difix(
                        conditioning,
                        prompt_tokens=batch["input_ids"],
                        wrapped_reference_validity=(
                            wrapped_reference_validity
                        ),
                    )
                    (
                        base_l2,
                        prediction_flat,
                        target_flat,
                        loss_row_weights,
                    ) = _wrapped_loss_tensors(
                        prediction,
                        target,
                        wrapped_reference_validity,
                        args.wrapped_reference_loss_mask,
                    )

                    loss_l2 = base_l2 * args.lambda_l2
                    lpips_rows = net_lpips(
                        prediction_flat.float(), target_flat.float()
                    ).reshape(-1)
                    loss_lpips = (
                        (
                            lpips_rows
                            * loss_row_weights.to(lpips_rows)
                        ).sum()
                        / loss_row_weights.sum().clamp_min(1.0)
                        * args.lambda_lpips
                    )
                    loss = loss_l2 + loss_lpips
                    loss_gram = torch.zeros(
                        (), device=conditioning.device, dtype=torch.float32
                    )

                    if (
                        args.lambda_gram > 0
                        and global_step > args.gram_loss_warmup_steps
                    ):
                        if not net_vgg_on_device:
                            net_vgg.to(conditioning.device)
                            net_vgg_on_device = True
                        crop_height = min(args.gram_crop_size, height)
                        crop_width = min(args.gram_crop_size, width)
                        top = random.randint(0, height - crop_height)
                        left = random.randint(0, width - crop_width)

                        prediction_vgg = imagenet_normalize(
                            prediction_flat * 0.5 + 0.5
                        )
                        prediction_vgg = crop(
                            prediction_vgg,
                            top,
                            left,
                            crop_height,
                            crop_width,
                        )
                        target_vgg = imagenet_normalize(
                            target_flat * 0.5 + 0.5
                        )
                        target_vgg = crop(
                            target_vgg,
                            top,
                            left,
                            crop_height,
                            crop_width,
                        )
                        loss_gram = (
                            gram_loss(
                                prediction_vgg[
                                    loss_row_weights > 0
                                ].float(),
                                target_vgg[
                                    loss_row_weights > 0
                                ].float(),
                                net_vgg,
                            )
                            * args.lambda_gram
                        )
                        loss = loss + loss_gram

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        layers_to_optimize, args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress_bar.update(1)
            logs = {
                "loss": _gather_mean(accelerator, loss),
                "opt_step": 1,
                "grad_norm": _gather_mean(accelerator, grad_norm),
                "l2": _gather_mean(accelerator, loss_l2),
                "lpips": _gather_mean(accelerator, loss_lpips),
                "gram": _gather_mean(accelerator, loss_gram),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if override_info is not None:
                logs.update(
                    {
                        "oracle_override_patch_fraction": _gather_mean(
                            accelerator,
                            override_info["selected_fraction"],
                        ),
                        "oracle_eligible_patch_fraction": _gather_mean(
                            accelerator,
                            override_info["eligible_fraction"],
                        ),
                    }
                )
                logs["oracle_selected_of_eligible"] = _gather_ratio(
                    accelerator,
                    override_info["selected_patch_count"],
                    override_info["eligible_patch_count"],
                )
            progress_bar.set_postfix(
                loss=f"{logs['loss']:.4f}",
                lpips=f"{logs['lpips']:.4f}",
            )

            training_panels = []
            if accelerator.is_main_process and _interval_due(
                global_step, args.viz_freq
            ):
                training_panels = [
                    _image_panel(
                        conditioning,
                        target,
                        prediction,
                        index,
                        str(batch["record_id"][index]),
                        wrapped_reference_validity,
                        original_conditioning=original_conditioning,
                        render_override_mask=(
                            override_info["copy_mask"]
                            if override_info is not None
                            else None
                        ),
                    )
                    for index in range(batch_size)
                ]

            if _interval_due(global_step, args.checkpointing_steps):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    output_path = os.path.join(
                        args.output_dir,
                        "checkpoints",
                        f"model_{global_step}.pkl",
                    )
                    save_ckpt(
                        accelerator.unwrap_model(net_difix),
                        optimizer,
                        output_path,
                        lr_scheduler=lr_scheduler,
                        global_step=global_step,
                        training_args=dict(vars(args)),
                    )
                accelerator.wait_for_everyone()

            if _interval_due(global_step, args.eval_freq):
                accelerator.wait_for_everyone()
                _run_validation(
                    accelerator,
                    net_difix,
                    dataloader_val,
                    weight_dtype,
                    args.num_samples_eval,
                    args.num_validation_panel_samples,
                    args.output_dir,
                    global_step,
                    args,
                )
                accelerator.wait_for_everyone()

            if accelerator.is_main_process and args.report_to != "none":
                accelerator.log(logs, step=global_step)
                for index, panel in enumerate(training_panels):
                    for tracker in accelerator.trackers:
                        try:
                            _log_image(
                                tracker,
                                f"train/panels/pair_{index:02d}",
                                panel,
                                global_step,
                            )
                        except Exception as exc:
                            print(
                                "Tracker training-panel upload failed "
                                f"({tracker.name}, pair {index}): {exc}"
                            )

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if (
        accelerator.is_main_process
        and args.save_final_checkpoint
        and global_step > 0
    ):
        output_path = os.path.join(
            args.output_dir, "checkpoints", f"model_{global_step}.pkl"
        )
        save_ckpt(
            accelerator.unwrap_model(net_difix),
            optimizer,
            output_path,
            lr_scheduler=lr_scheduler,
            global_step=global_step,
            training_args=dict(vars(args)),
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the original two-view, reference-conditioned Difix model "
            "on an image-pair manifest."
        )
    )

    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)
    parser.add_argument("--gram_crop_size", default=400, type=int)

    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--validation_dataset_path", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--validation_split", default="test")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--validation_data_root", default=None)
    parser.add_argument("--path_prefix_from", default=None)
    parser.add_argument("--path_prefix_to", default=None)
    parser.add_argument(
        "--storage_backend",
        default="local",
        choices=["local", "s3_stream"],
    )
    parser.add_argument("--s3_cache_dir", default=None)
    parser.add_argument(
        "--s3_cache_images",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--s3_max_pool_connections",
        default=32,
        type=int,
    )
    parser.add_argument("--s3_tar_routes_path", default=None)
    parser.add_argument(
        "--wrapped_reference_cache_dir",
        default=None,
        help=(
            "PixelDiT-compatible wrapped-reference .npz cache. Providing "
            "this enables the third wrapped-reference view."
        ),
    )
    parser.add_argument(
        "--validation_wrapped_reference_cache_dir",
        default=None,
        help=(
            "Optional validation cache; defaults to "
            "--wrapped_reference_cache_dir."
        ),
    )
    parser.add_argument(
        "--wrapped_reference_fill",
        default="black",
        choices=["black", "zero", "corrupt"],
    )
    parser.add_argument(
        "--wrapped_reference_builder_repo",
        default=None,
        help=(
            "Optional PixelDiT repository used to build missing wrapped "
            "cache entries online."
        ),
    )
    parser.add_argument(
        "--wrapped_reference_depth_cache_dir",
        default=None,
        help=(
            "Variant-B geometry/depth cache used by the online wrapped "
            "reference builder."
        ),
    )
    parser.add_argument(
        "--validation_wrapped_reference_depth_cache_dir",
        default=None,
        help=(
            "Optional validation Variant-B cache; defaults to "
            "--wrapped_reference_depth_cache_dir."
        ),
    )
    parser.add_argument(
        "--wrapped_reference_splat_radius",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--cache_built_wrapped_references",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Persist wrapped references generated by the online builder. "
            "Disable for large streamed corpora to avoid an unbounded RGB "
            "cache; source depth entries remain cached."
        ),
    )
    parser.add_argument(
        "--wrapped_reference_loss_mask",
        default="valid_pixels",
        choices=["all", "valid_pixels"],
        help=(
            "Scope of wrapped-view supervision. Attention validity remains "
            "independently controlled by the patch-fraction threshold."
        ),
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--default_prompt", default="remove degradation")
    parser.add_argument("--height", default=576, type=int)
    parser.add_argument("--width", default=1024, type=int)

    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--num_samples_eval", default=100, type=int)
    parser.add_argument("--num_validation_panel_samples", default=10, type=int)
    parser.add_argument("--validation_num_workers", default=0, type=int)
    parser.add_argument("--viz_freq", default=100, type=int)
    parser.add_argument("--tracker_project_name", default="difix")
    parser.add_argument("--tracker_run_name", required=True)

    parser.add_argument(
        "--pretrained_model_name_or_path", default="stabilityai/sd-turbo"
    )
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument(
        "--mv_unet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the official two-view joint-attention U-Net (required here).",
    )
    parser.add_argument(
        "--sparse_mv_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use split-query sparse attn1 routing for render, reference, and "
            "wrapped-reference views."
        ),
    )
    parser.add_argument(
        "--view_attention_query_modes",
        default="all,all,valid_only",
        help=(
            "Comma-separated query modes for render, reference, wrapped: "
            "all|valid_only|none."
        ),
    )
    parser.add_argument(
        "--view_attention_matrix",
        default=(
            "all,all,valid_only;"
            "all,all,valid_only;"
            "all,all,valid_only"
        ),
        help=(
            "Semicolon-separated query rows and comma-separated source "
            "columns. Each cell is all|valid_only|none."
        ),
    )
    parser.add_argument(
        "--wrapped_reference_patch_valid_fraction",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--wrapped_reference_denoise_mask",
        default="all",
        choices=["all", "valid_only"],
        help=(
            "Whether the one-step scheduler update changes all wrapped "
            "latents or only valid wrapped latent cells."
        ),
    )
    parser.add_argument(
        "--render_endpoint_mode",
        default="original",
        choices=["original", "wrapped_patch_override"],
        help=(
            "Optionally replace selected corrupt-render pixels with the "
            "pose-aligned wrapped reference before the SD-Turbo VAE."
        ),
    )
    parser.add_argument(
        "--render_endpoint_override_policy",
        default="all_valid",
        choices=["all_valid", "oracle_gt"],
        help=(
            "Choose all geometrically eligible wrapped patches or retain "
            "only patches preferred by a clean-GT oracle."
        ),
    )
    parser.add_argument(
        "--render_endpoint_override_patch_size",
        default=16,
        type=int,
        help="Pixel-space side length of each replacement patch.",
    )
    parser.add_argument(
        "--render_endpoint_override_valid_fraction",
        default=0.95,
        type=float,
        help=(
            "Fraction of valid pixels required for oracle replacement. "
            "This is independent of sparse-attention validity."
        ),
    )
    parser.add_argument(
        "--render_endpoint_oracle_metric",
        default="lpips",
        choices=["l1", "mse", "lpips", "lpips_3x3"],
    )
    parser.add_argument(
        "--render_endpoint_oracle_window_patches",
        default=3,
        type=int,
        help=(
            "Odd context-window side length in patches for LPIPS. "
            "lpips_3x3 forces this value to three."
        ),
    )
    parser.add_argument(
        "--render_endpoint_oracle_margin",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--render_endpoint_oracle_relative_gain",
        default=0.0,
        type=float,
        help=(
            "Require wrapped_error < render_error * (1 - relative_gain)."
        ),
    )
    parser.add_argument(
        "--render_endpoint_oracle_lpips_chunk_size",
        default=64,
        type=int,
    )

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--max_train_steps", default=10_000, type=int)
    parser.add_argument("--checkpointing_steps", default=1000, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--lr_warmup_steps", default=500, type=int)
    parser.add_argument("--lr_num_cycles", default=1, type=int)
    parser.add_argument("--lr_power", default=1.0, type=float)
    parser.add_argument("--dataloader_num_workers", default=8, type=int)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--adam_weight_decay", default=1e-2, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--report_to",
        default="wandb",
        choices=["wandb", "tensorboard", "clearml", "all", "none"],
    )
    parser.add_argument(
        "--mixed_precision",
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument(
        "--save_final_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--validate_on_start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run validation immediately after initialization or checkpoint "
            "resume, before the first training batch."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
