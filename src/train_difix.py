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


PANEL_TITLES = (
    "corrupt",
    "reference",
    "clean",
    "restored render",
    "restored reference",
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


def _image_panel(conditioning, target, prediction, index, caption):
    row = {
        "panels": [
            to_uint8_panel(conditioning[index, 0]),
            to_uint8_panel(conditioning[index, 1]),
            to_uint8_panel(target[index, 0]),
            to_uint8_panel(prediction[index, 0]),
            to_uint8_panel(prediction[index, 1]),
        ],
        "caption": caption,
    }
    return build_labeled_grid(
        [row],
        PANEL_TITLES,
    )


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


def _dataset_kwargs(args, data_root=None):
    return {
        "height": args.height,
        "width": args.width,
        "prompt_override": args.prompt,
        "default_prompt": args.default_prompt,
        "require_reference": True,
        "data_root": args.data_root if data_root is None else data_root,
        "path_prefix_from": args.path_prefix_from,
        "path_prefix_to": args.path_prefix_to,
    }


def _build_validation_loader(args, tokenizer):
    validation_path = args.validation_dataset_path or args.dataset_path
    dataset = PairedDataset(
        dataset_path=validation_path,
        split=args.validation_split,
        tokenizer=tokenizer,
        **_dataset_kwargs(
            args,
            data_root=args.validation_data_root or args.data_root,
        ),
    )
    if len(dataset) == 0:
        raise ValueError(
            f"Validation split {args.validation_split!r} in "
            f"{validation_path} is empty"
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
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
):
    model = accelerator.unwrap_model(net_difix)
    model.eval()
    local_rows = []
    row_dir = os.path.join(output_dir, "eval", f"step_{step:09d}")
    if accelerator.is_main_process:
        os.makedirs(row_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    for sample_index, batch in enumerate(dataloader):
        if sample_index >= num_samples:
            break
        if sample_index % accelerator.num_processes != accelerator.process_index:
            continue

        conditioning = batch["conditioning_pixel_values"].to(
            accelerator.device, dtype=weight_dtype
        )
        target = batch["output_pixel_values"].to(
            accelerator.device, dtype=weight_dtype
        )
        input_ids = batch["input_ids"].to(accelerator.device)

        with torch.no_grad(), accelerator.autocast():
            prediction = model(conditioning, prompt_tokens=input_ids)
        render_metrics = batch_metrics(prediction[:, 0], target[:, 0])
        input_metrics = batch_metrics(conditioning[:, 0], target[:, 0])
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

        record_id = batch["record_id"][0]
        row_path = None
        if sample_index < num_panel_samples:
            panel = _image_panel(
                conditioning,
                target,
                prediction,
                0,
                str(record_id),
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
            }
        )

    rows = sorted(
        _gather_rows(accelerator, local_rows),
        key=lambda row: row["index"],
    )
    average_metrics, median_metrics, metric_counts = _aggregate_metric_rows(
        rows
    )

    if not rows:
        raise ValueError("Validation produced no samples")

    if accelerator.is_main_process:
        summary_path = os.path.join(row_dir, "validation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "namespace": "restore-val",
                    "step": step,
                    "num_pairs": len(rows),
                    "columns": list(PANEL_TITLES),
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
        image_tag = "restore-val/" + " | ".join(PANEL_TITLES)
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
        for prefix in ("input", "ours_ref", "restored_reference"):
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
        **_dataset_kwargs(args),
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
        _build_validation_loader(args, net_difix.tokenizer)
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
    net_vgg.to(accelerator.device)

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
                batch_size, num_views, _, height, width = conditioning.shape

                with accelerator.autocast():
                    prediction = net_difix(
                        conditioning, prompt_tokens=batch["input_ids"]
                    )
                    target_flat = rearrange(
                        target, "b v c h w -> (b v) c h w"
                    )
                    prediction_flat = rearrange(
                        prediction, "b v c h w -> (b v) c h w"
                    )

                    loss_l2 = (
                        F.mse_loss(
                            prediction_flat.float(),
                            target_flat.float(),
                            reduction="mean",
                        )
                        * args.lambda_l2
                    )
                    loss_lpips = (
                        net_lpips(
                            prediction_flat.float(), target_flat.float()
                        ).mean()
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
                                prediction_vgg.float(),
                                target_vgg.float(),
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
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
