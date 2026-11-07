import hashlib
import importlib
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


class IndexedJsonlRecords:
    """Random-access JSONL backed by a shared, memory-mapped uint64 index."""

    def __init__(self, path):
        self.path = Path(path)
        self.index_path = Path(f"{self.path}.idx")
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"Missing JSONL index {self.index_path}. Create it with "
                "scripts/build_difix_manifest.py."
            )
        index_size = self.index_path.stat().st_size
        if index_size % np.dtype("<u8").itemsize != 0:
            raise ValueError(
                f"Invalid JSONL index size for {self.index_path}: {index_size}"
            )
        self.length = index_size // np.dtype("<u8").itemsize
        self.offsets = None
        self.file_handle = None
        self.file_handle_pid = None
        self._open_offsets()

    def _open_offsets(self):
        if self.length:
            self.offsets = np.memmap(
                self.index_path, dtype="<u8", mode="r", shape=(self.length,)
            )
        else:
            self.offsets = np.empty((0,), dtype="<u8")

    def __len__(self):
        return self.length

    def _get_file_handle(self):
        current_pid = os.getpid()
        if self.file_handle is None or self.file_handle_pid != current_pid:
            if self.file_handle is not None:
                self.file_handle.close()
            self.file_handle = self.path.open("rb")
            self.file_handle_pid = current_pid
        return self.file_handle

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)

        file_handle = self._get_file_handle()
        file_handle.seek(int(self.offsets[idx]))
        line = file_handle.readline()
        if not line:
            raise RuntimeError(
                f"Index {idx} in {self.index_path} points past the JSONL data"
            )
        record = json.loads(line)
        record_id = str(
            record.pop("record_id", record.get("pair_id", record.get("id", idx)))
        )
        return record_id, record

    def __getstate__(self):
        state = self.__dict__.copy()
        state["offsets"] = None
        state["file_handle"] = None
        state["file_handle_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_offsets()

    def __del__(self):
        if self.file_handle is not None:
            self.file_handle.close()


class PairedDataset(torch.utils.data.Dataset):
    """Two-view Difix dataset: corrupt render + rank-0 reference."""

    def __init__(
        self,
        dataset_path,
        split,
        height=576,
        width=1024,
        tokenizer=None,
        prompt_override=None,
        default_prompt="remove degradation",
        require_reference=True,
        data_root=None,
        path_prefix_from=None,
        path_prefix_to=None,
        wrapped_reference_cache_dir=None,
        wrapped_reference_fill="black",
        wrapped_reference_builder_repo=None,
        wrapped_reference_depth_cache_dir=None,
        wrapped_reference_splat_radius=1,
    ):
        super().__init__()

        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.record_store = None
        self.records = None
        if self.dataset_path.suffix.lower() in {".jsonl", ".ndjson"}:
            self.record_store = IndexedJsonlRecords(self.dataset_path)
            self.img_ids = None
        else:
            if self.dataset_path.stat().st_size > 2 * 1024**3:
                raise ValueError(
                    f"{self.dataset_path} is too large for per-process "
                    "json.load(). Convert it with "
                    "scripts/build_difix_manifest.py and pass the resulting "
                    ".jsonl file instead."
                )
            with self.dataset_path.open("r") as f:
                manifest = json.load(f)

            if split not in manifest:
                available = ", ".join(sorted(manifest.keys()))
                raise KeyError(
                    f"Split {split!r} is not present in {self.dataset_path}. "
                    f"Available top-level keys: {available}"
                )

            split_data = manifest[split]
            if isinstance(split_data, dict):
                self.records = list(split_data.items())
            elif isinstance(split_data, list):
                self.records = [
                    (str(record.get("pair_id", record.get("id", idx))), record)
                    for idx, record in enumerate(split_data)
                ]
            else:
                raise TypeError(
                    f"Manifest split {split!r} must be a dict or list, got "
                    f"{type(split_data).__name__}"
                )
            self.img_ids = [record_id for record_id, _ in self.records]
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.prompt_override = prompt_override
        self.default_prompt = default_prompt
        self.require_reference = require_reference
        self.wrapped_reference_cache_dir = (
            Path(wrapped_reference_cache_dir).expanduser()
            if wrapped_reference_cache_dir is not None
            else None
        )
        self.wrapped_reference_fill = str(wrapped_reference_fill)
        if self.wrapped_reference_fill not in ("black", "zero", "corrupt"):
            raise ValueError(
                "wrapped_reference_fill must be black|zero|corrupt"
            )
        self.wrapped_reference_builder_repo = (
            Path(wrapped_reference_builder_repo).expanduser().resolve()
            if wrapped_reference_builder_repo is not None
            else None
        )
        self.wrapped_reference_depth_cache_dir = (
            Path(wrapped_reference_depth_cache_dir).expanduser().resolve()
            if wrapped_reference_depth_cache_dir is not None
            else None
        )
        self.wrapped_reference_splat_radius = int(
            wrapped_reference_splat_radius
        )
        if self.wrapped_reference_splat_radius < 0:
            raise ValueError(
                "wrapped_reference_splat_radius must be non-negative"
            )
        if (
            self.wrapped_reference_builder_repo is None
        ) != (
            self.wrapped_reference_depth_cache_dir is None
        ):
            raise ValueError(
                "wrapped_reference_builder_repo and "
                "wrapped_reference_depth_cache_dir must be provided together"
            )
        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else self.dataset_path.parent
        )

        if (path_prefix_from is None) != (path_prefix_to is None):
            raise ValueError(
                "path_prefix_from and path_prefix_to must be provided together"
            )
        self.path_prefix_from = (
            os.path.expanduser(path_prefix_from)
            if path_prefix_from is not None
            else None
        )
        self.path_prefix_to = (
            os.path.expanduser(path_prefix_to)
            if path_prefix_to is not None
            else None
        )

    def __len__(self):
        if self.record_store is not None:
            return len(self.record_store)
        return len(self.records)

    def _resolve_path(self, raw_path):
        path = os.path.expanduser(str(raw_path))
        if self.path_prefix_from is not None and path.startswith(
            self.path_prefix_from
        ):
            path = self.path_prefix_to + path[len(self.path_prefix_from) :]

        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.data_root / resolved
        return resolved

    def _load_image(self, raw_path, field_name, record_id):
        path = self._resolve_path(raw_path)
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                # Match PixelDiT preprocessing exactly: resize the PIL image
                # before converting it to a tensor. Resizing the float tensor
                # produces slightly different pixels.
                image = F.resize(
                    image,
                    self.image_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                tensor = F.to_tensor(image)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {field_name} for record {record_id!r}: {path}"
            ) from exc

        return F.normalize(tensor, mean=[0.5], std=[0.5])

    @staticmethod
    def _sha1_cache_key(clean_path, corrupt_path):
        raw = f"{clean_path}|{corrupt_path}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _wrapped_reference_cache_candidates(self, record_id, record):
        explicit_key = record.get("wrapped_reference_cache_key")
        keys = []
        if explicit_key:
            keys.append(str(explicit_key))
        keys.append(
            self._sha1_cache_key(
                record["target_image"],
                record["image"],
            )
        )
        keys.append(
            self._sha1_cache_key(
                self._resolve_path(record["target_image"]),
                self._resolve_path(record["image"]),
            )
        )
        resolved_clean = str(self._resolve_path(record["target_image"]))
        resolved_corrupt = str(self._resolve_path(record["image"]))
        if (
            self.path_prefix_from is not None
            and self.path_prefix_to is not None
            and resolved_clean.startswith(self.path_prefix_to)
            and resolved_corrupt.startswith(self.path_prefix_to)
        ):
            keys.append(
                self._sha1_cache_key(
                    self.path_prefix_from
                    + resolved_clean[len(self.path_prefix_to) :],
                    self.path_prefix_from
                    + resolved_corrupt[len(self.path_prefix_to) :],
                )
            )
        keys.append(str(record_id))

        candidates = []
        seen = set()
        for key in keys:
            for path in (
                self.wrapped_reference_cache_dir
                / key[:2]
                / f"{key}.npz",
                self.wrapped_reference_cache_dir / f"{key}.npz",
            ):
                if path not in seen:
                    candidates.append(path)
                    seen.add(path)
        return candidates

    def _build_wrapped_reference(
        self,
        record_id,
        record,
        corrupt,
        reference,
    ):
        camera = record.get("camera")
        if not camera:
            raise RuntimeError(
                f"Record {record_id!r} cannot build a wrapped reference "
                "online because it has no camera metadata"
            )

        repo = self.wrapped_reference_builder_repo
        for path in (
            repo,
            repo / "PixelDiT",
            repo / "PixelDiT" / "t2i",
        ):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        try:
            warped_module = importlib.import_module(
                "restore.warped_reference"
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the PixelDiT wrapped-reference builder "
                f"from {repo}"
            ) from exc

        resolved_clean = str(self._resolve_path(record["target_image"]))
        resolved_corrupt = str(self._resolve_path(record["image"]))
        resolved_reference = str(
            self._resolve_path(record["ref_image"])
        )
        pixeldit_record = {
            "clean": resolved_clean,
            "corrupted": resolved_corrupt,
            "geometry_corrupted": record.get(
                "geometry_corrupted",
                record["image"],
            ),
            "camera": camera,
            "geometry_path_map": {
                resolved_reference: record.get(
                    "geometry_ref_image",
                    record["ref_image"],
                ),
            },
        }
        resize = T.Compose(
            [
                T.Resize(
                    self.image_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            ]
        )
        # The PixelDiT builder intentionally returns an all-invalid warp when
        # its depth lookup misses. That behavior is useful at inference but
        # dangerous while populating a training cache because a remapped path
        # or wrong cache root would be persisted as a valid empty result.
        # Resolve the required depth entry explicitly so configuration errors
        # fail before any .npz file is written. A real no-overlap view may
        # still produce an all-invalid warp after this check.
        geometry_map = pixeldit_record.get("geometry_path_map") or {}
        source_reference = os.path.normpath(
            str(geometry_map.get(resolved_reference, resolved_reference))
        )
        source_clean = os.path.normpath(
            str(geometry_map.get(resolved_clean, resolved_clean))
        )
        if source_reference != source_clean:
            warped_module.load_omnivggt_depth_window(
                pixeldit_record,
                resolved_reference,
                resize,
                (0, 0, *self.image_size),
                str(self.wrapped_reference_depth_cache_dir),
            )
        with torch.inference_mode():
            wrapped, validity = warped_module.build_warped_reference(
                pixeldit_record,
                [resolved_reference],
                [reference],
                corrupt,
                resize,
                (0, 0, *self.image_size),
                str(self.wrapped_reference_depth_cache_dir),
                splat_radius=self.wrapped_reference_splat_radius,
                fill="zero",
            )

        key = self._sha1_cache_key(
            resolved_clean,
            resolved_corrupt,
        )
        path = self.wrapped_reference_cache_dir / key[:2] / f"{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = (
            (wrapped.float().clamp(-1, 1) + 1.0)
            .mul(127.5)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        valid = validity.bool().reshape(-1).cpu().numpy()
        rgb.reshape(-1, 3)[~valid] = 0
        temporary = path.with_name(
            f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                rgb=rgb,
                validity=np.packbits(valid, bitorder="little"),
                shape=np.asarray(self.image_size, dtype=np.int32),
            )
        os.replace(temporary, path)
        return path

    def _load_wrapped_reference(
        self,
        record_id,
        record,
        corrupt,
        reference,
    ):
        candidates = self._wrapped_reference_cache_candidates(
            record_id, record
        )
        path = next((path for path in candidates if path.is_file()), None)
        if (
            path is None
            and self.wrapped_reference_builder_repo is not None
        ):
            path = self._build_wrapped_reference(
                record_id,
                record,
                corrupt,
                reference,
            )
        if path is None:
            attempted = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"Missing wrapped-reference cache entry for {record_id!r}; "
                f"tried: {attempted}"
            )

        with np.load(path, allow_pickle=False) as data:
            required = {"rgb", "validity", "shape"}
            if not required.issubset(data.files):
                raise RuntimeError(
                    f"Invalid wrapped-reference cache entry {path}; "
                    f"expected keys {sorted(required)}"
                )
            rgb = np.asarray(data["rgb"], dtype=np.uint8)
            height, width = (int(value) for value in data["shape"])
            validity = np.unpackbits(
                np.asarray(data["validity"], dtype=np.uint8),
                count=height * width,
                bitorder="little",
            ).reshape(height, width)

        if rgb.shape != (height, width, 3):
            raise RuntimeError(
                f"Invalid wrapped-reference RGB shape {rgb.shape} in {path}"
            )
        if (height, width) != self.image_size:
            raise RuntimeError(
                "Wrapped-reference cache resolution must match the training "
                f"resolution exactly: cache={(height, width)}, "
                f"training={self.image_size}, path={path}"
            )

        wrapped_image = Image.fromarray(rgb, mode="RGB")
        wrapped = F.normalize(
            F.to_tensor(wrapped_image),
            mean=[0.5],
            std=[0.5],
        )

        validity_image = Image.fromarray(
            validity.astype(np.uint8) * 255,
            mode="L",
        )
        validity_tensor = (
            F.pil_to_tensor(validity_image).float().div(255.0) > 0.5
        )

        if self.wrapped_reference_fill == "black":
            fallback = torch.full_like(wrapped, -1.0)
        elif self.wrapped_reference_fill == "zero":
            fallback = torch.zeros_like(wrapped)
        else:
            fallback = corrupt
        wrapped = torch.where(validity_tensor, wrapped, fallback)
        return wrapped, validity_tensor.float(), str(path)

    def __getitem__(self, idx):
        if self.record_store is not None:
            record_id, record = self.record_store[idx]
        else:
            record_id, record = self.records[idx]

        for required_key in ("image", "target_image"):
            if required_key not in record:
                raise KeyError(
                    f"Record {record_id!r} is missing required field "
                    f"{required_key!r}"
                )

        ref_path = record.get("ref_image")
        if self.require_reference and not ref_path:
            raise KeyError(
                f"Record {record_id!r} is missing required rank-0 reference "
                "field 'ref_image'"
            )

        corrupt = self._load_image(record["image"], "image", record_id)
        clean = self._load_image(
            record["target_image"], "target_image", record_id
        )

        if ref_path:
            reference = self._load_image(ref_path, "ref_image", record_id)
            conditioning_views = [corrupt, reference]
            target_views = [clean, reference]
            wrapped_validity = None
            wrapped_cache_path = None
            if self.wrapped_reference_cache_dir is not None:
                (
                    wrapped_reference,
                    wrapped_validity,
                    wrapped_cache_path,
                ) = self._load_wrapped_reference(
                    record_id,
                    record,
                    corrupt,
                    reference,
                )
                conditioning_views.append(wrapped_reference)
                target_views.append(clean)
            conditioning = torch.stack(conditioning_views, dim=0)
            target = torch.stack(target_views, dim=0)
        else:
            conditioning = corrupt.unsqueeze(0)
            target = clean.unsqueeze(0)
            wrapped_validity = None
            wrapped_cache_path = None

        caption = (
            self.prompt_override
            if self.prompt_override is not None
            else record.get("prompt") or self.default_prompt
        )
        out = {
            "output_pixel_values": target,
            "conditioning_pixel_values": conditioning,
            "caption": caption,
            "record_id": record_id,
        }
        if wrapped_validity is not None:
            out["wrapped_reference_validity"] = wrapped_validity
            out["wrapped_reference_cache_path"] = wrapped_cache_path

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids
            # Return [sequence], so DataLoader collates to [batch, sequence].
            out["input_ids"] = input_ids.squeeze(0)

        return out
