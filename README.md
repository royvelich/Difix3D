# Difix3D+

**Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models**  
[Jay Zhangjie Wu*](https://zhangjiewu.github.io/), [Yuxuan Zhang*](https://scholar.google.com/citations?user=Jt5VvNgAAAAJ&hl=en), [Haithem Turki](https://haithemturki.com/), [Xuanchi Ren](https://xuanchiren.com/), [Jun Gao](https://www.cs.toronto.edu/~jungao/),  
[Mike Zheng Shou](https://sites.google.com/view/showlab/home?authuser=0), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Zan Gojcic†](https://zgojcic.github.io/), [Huan Ling†](https://www.cs.toronto.edu/~linghuan/) _(*/† equal contribution/advising)_  
CVPR 2025 (Oral)  
[Project Page](https://research.nvidia.com/labs/toronto-ai/difix3d/) | [Paper](https://arxiv.org/abs/2503.01774) | [Model](https://huggingface.co/nvidia/difix) | [Demo](https://huggingface.co/spaces/nvidia/difix)

<div align="center">
  <img src="assets/demo.gif" alt=""  width="1100" />
</div>


## News

* [11/06/2025] Code and models are now available! We will present our work at CVPR 2025 ([oral](https://cvpr.thecvf.com/virtual/2025/oral/35364), [poster](https://cvpr.thecvf.com/virtual/2025/poster/34172)). See you in Nashville🎵!


## Setup

```bash
git clone https://github.com/nv-tlabs/Difix3D.git
cd Difix3D
pip install -r requirements.txt
```

## Quickstart (diffusers)

```
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

input_image = load_image("assets/example_input.png")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("example_output.png")
```

Optionally, you can use a reference image to guide the denoising process.
```
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
pipe.to("cuda")

input_image = load_image("assets/example_input.png")
ref_image = load_image("assets/example_ref.png")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("example_output.png")
```

## Difix: Single-step diffusion for 3D artifact removal

### Training

#### Data Preparation

Prepare your dataset in the following JSON format:

```json
{
  "train": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "ref_image": "{PATH_TO_REF_IMAGE}",
      "prompt": "remove degradation"
    }
  },
  "test": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "ref_image": "{PATH_TO_REF_IMAGE}",
      "prompt": "remove degradation"
    }
  }
}
```

The loader accepts either dictionaries or lists for `train` and `test`, and
ignores additional metadata fields. For the vanilla reference-conditioned
experiment it always uses `ref_image` (our rank-0 reference):

| Stream | Model input | Supervision target |
| --- | --- | --- |
| Render | corrupt render (`image`) | clean render (`target_image`) |
| Reference | rank-0 reference (`ref_image`) | the same rank-0 reference |

This is the original two-view Difix objective. It does not use depth, warps,
wrapped references, validity masks, or custom attention.

Our full training `data.json` is too large to load once per distributed
process. Convert its `train` split to compact indexed JSONL once:

```bash
python scripts/build_difix_manifest.py \
    --input /path/to/training-corpus/data.json \
    --output /path/to/training-corpus/difix_train.jsonl \
    --split train
```

This keeps only `image`, `target_image`, `ref_image`, and `prompt`, and creates
`difix_train.jsonl.idx` for random access. The index is memory-mapped, so all
training and DataLoader processes can share it without each parsing the full
JSON manifest. Pass `difix_train.jsonl` as `--dataset_path`; regular JSON
manifests remain supported for smaller validation sets.

#### Single GPU

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/difix/train \
    --dataset_path="data/difix_train.jsonl" \
    --max_train_steps 10000 \
    --height=576 --width=1024 --learning_rate 2e-5 \
    --train_batch_size=1 --dataloader_num_workers 8 \
    --mv_unet \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=1000 --eval_freq 1000 --viz_freq 100 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "train" --timestep 199
```

#### Multipe GPUs

```bash
export NUM_NODES=1
export NUM_GPUS=8
accelerate launch --mixed_precision=bf16 --main_process_port 29501 --multi_gpu --num_machines $NUM_NODES --num_processes $NUM_GPUS src/train_difix.py \
    --output_dir=./outputs/difix/train \
    --dataset_path="data/difix_train.jsonl" \
    --validation_dataset_path="data/validation_data.json" \
    --validation_data_root="data/validation_images" \
    --max_train_steps 10000 \
    --height=576 --width=1024 --learning_rate 2e-5 \
    --train_batch_size=1 --dataloader_num_workers 8 \
    --mv_unet \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=1000 --eval_freq 1000 --viz_freq 100 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
    --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "train" --timestep 199
```

Use `--validation_dataset_path` when training and validation are stored in
different manifests. Relative training paths are resolved against the training
manifest directory (or `--data_root`), while relative validation paths use the
validation manifest directory (or `--validation_data_root`, falling back to
`--data_root`). Machine-specific absolute path prefixes can be remapped with
`--path_prefix_from OLD --path_prefix_to NEW`.

#### PixelDiT-compatible validation and ClearML

Install the pinned validation backends without changing Difix's pinned
Transformers version:

```bash
scripts/install_validation_deps.sh /path/to/training/venv/bin/python
```

Then pass `--report_to=clearml`. Training logs `loss`, `opt_step`,
`grad_norm`, every component loss, and `lr` at each optimizer step.
Validation logs exact per-pair means and medians under
`restore-val/metrics/...` for the same PyIQA metrics as PixelDiT: PSNR, color
SSIM, LPIPS-VGG, TOPIQ-FR, DISTS, and ST-LPIPS. `ours_ref_*` scores the
restored render against clean GT and the mandatory `input_*` baseline scores
the untouched corrupt render. Labeled per-pair panels are saved under
`OUTPUT_DIR/eval/step_NNNNNNNNN/` and uploaded to ClearML.

### Inference

Place the `model_*.pkl` in the `checkpoints` directory. You can run inference using the following command:

```bash
python src/inference_difix.py \
    --model_path "checkpoints/model.pkl" \
    --input_image "assets/example_input.png" \
    --prompt "remove degradation" \
    --output_dir "outputs/difix" \
    --timestep 199
```


## Difix3D: Progressive 3D update

### Data Format

The data should be organized in the following structure:

```
DATA_DIR/
├── {SCENE_ID}
│   ├── colmap
│   │   ├── sparse
│   │   │   └── 0
│   │   │       ├── cameras.bin
│   │   │       ├── database.db
│   │   │       └── ...
│   ├── images
│   │   ├── image_train_000001.png
│   │   ├── image_train_000002.png
│   │   ├── ...
│   │   ├── image_eval_000200.png
│   │   ├── image_eval_000201.png
│   │   └── ...
│   ├── images_2
│   ├── images_4
│   └── images_8
```

### nerfstudio

Setup the nerfstudio environment.
```bash
cd examples/nerfstudio
pip install -e .
cd ../..
```

Run Difix3D finetuning with nerfstudio.
```bash
SCENE_ID=032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7
DATA=DATA_DIR/${SCENE_ID}
DATA_FACTOR=4
CKPT_PATH=CKPR_DIR/${SCENE_ID}/nerfacto/nerfstudio_models/step-000029999.ckpt # Path to the pretrained checkpoint file
OUTPUT_DIR=outputs/difix3d/nerfacto/${SCENE_ID}

CUDA_VISIBLE_DEVICES=0 ns-train difix3d \
    --data ${DATA} --pipeline.model.appearance-embed-dim 0 --pipeline.model.camera-optimizer.mode off --save_only_latest_checkpoint False --vis viewer \
    --output_dir ${OUTPUT_DIR} --experiment_name ${SCENE_ID} --timestamp '' --load-checkpoint ${CKPT_PATH} \
    --max_num_iterations 30000 --steps_per_eval_all_images 0 --steps_per_eval_batch 0 --steps_per_eval_image 0 --steps_per_save 2000 --viewer.quit-on-train-completion True \
    nerfstudio-data --orientation-method none --center_method none --auto-scale-poses False --downscale_factor ${DATA_FACTOR} --eval_mode filename
```

### gsplat

Install the gsplat following the instructions in the [gsplat repository](https://github.com/nerfstudio-project/gsplat?tab=readme-ov-file#installation).

Run Difix3D finetuning with gsplat.
```bash
SCENE_ID=032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7
DATA=DATA_DIR/${SCENE_ID}/gaussian_splat
DATA_FACTOR=4
CKPT_PATH=CKPT_DIR/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt # Path to the pretrained checkpoint file
OUTPUT_DIR=outputs/difix3d/gsplat/${SCENE_ID}

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA} --data_factor ${DATA_FACTOR} \
    --result_dir ${OUTPUT_DIR} --no-normalize-world-space --test_every 1 --ckpt ${CKPT_PATH}
```


## Difix3D+: With real-time post-rendering

Due to the limited capacity of reconstruction methods to represent sharp details, some regions remain blurry. To further enhance the novel views, we use our Difix model as the final post-processing step at render time.

```bash
python src/inference_difix.py \
    --model_path "checkpoints/model.pkl" \
    --input_image "PATH_TO_IMAGES" \
    --prompt "remove degradation" \
    --output_dir "outputs/difix3d+" \
    --timestep 199
```

## Acknowledgements

Our work is built upon the following projects:
- [diffusers](https://github.com/huggingface/diffusers)
- [img2img-turbo](https://github.com/GaParmar/img2img-turbo)
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [DL3DV-10K](https://github.com/DL3DV-10K/Dataset)
- [nerfbusters](https://github.com/ethanweber/nerfbusters)

Shoutout to all the contributors of these projects for their invaluable work that made this research possible.

## License/Terms of Use:

The use of the model and code is governed by the NVIDIA License. See [LICENSE.txt](LICENSE.txt) for details.
Additional Information:  [LICENSE.md · stabilityai/sd-turbo at main](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md)

## Citation

```bibtex
@inproceedings{wu2025difix3d+,
  title={DIFIX3D+: Improving 3D Reconstructions with Single-Step Diffusion Models},
  author={Wu, Jay Zhangjie and Zhang, Yuxuan and Turki, Haithem and Ren, Xuanchi and Gao, Jun and Shou, Mike Zheng and Fidler, Sanja and Gojcic, Zan and Ling, Huan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={26024--26035},
  year={2025}
}
```
