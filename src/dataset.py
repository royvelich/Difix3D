import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
            conditioning = torch.stack([corrupt, reference], dim=0)
            target = torch.stack([clean, reference], dim=0)
        else:
            conditioning = corrupt.unsqueeze(0)
            target = clean.unsqueeze(0)

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
