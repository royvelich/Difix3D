#!/usr/bin/env python3
"""Stream a large pair-corpus data.json into indexed, compact JSONL."""

import argparse
import json
import os
import struct
from pathlib import Path

try:
    import ijson
except ImportError as exc:
    raise SystemExit(
        "ijson is required. Install requirements.txt before running this script."
    ) from exc


REQUIRED_FIELDS = ("image", "target_image", "ref_image")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source data.json")
    parser.add_argument("--output", required=True, help="Output .jsonl path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--default_prompt", default="remove degradation")
    parser.add_argument("--progress_every", default=100_000, type=int)
    parser.add_argument(
        "--limit",
        default=0,
        type=int,
        help="Optional record limit for smoke tests; 0 converts the full split.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("streaming", "orjson"),
        default="streaming",
        help=(
            "Use the low-memory ijson stream or load the entire manifest with "
            "orjson for faster conversion on high-memory machines."
        ),
    )
    return parser.parse_args()


def iter_split_records(input_path, split, backend):
    if backend == "streaming":
        with input_path.open("rb") as source:
            yield from ijson.kvitems(source, split)
        return

    try:
        import orjson
    except ImportError as exc:
        raise RuntimeError(
            "The orjson backend requires orjson to be installed."
        ) from exc

    source_bytes = input_path.read_bytes()
    manifest = orjson.loads(source_bytes)
    del source_bytes
    if split not in manifest:
        raise KeyError(f"Split {split!r} is not present in {input_path}")

    split_data = manifest[split]
    if isinstance(split_data, dict):
        yield from split_data.items()
    elif isinstance(split_data, list):
        for index, record in enumerate(split_data):
            yield str(record.get("pair_id", record.get("id", index))), record
    else:
        raise TypeError(
            f"Manifest split {split!r} must be a dict or list, got "
            f"{type(split_data).__name__}"
        )


def main(args):
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    index_path = Path(f"{output_path}.idx")
    metadata_path = Path(f"{output_path}.meta.json")

    if input_path == output_path:
        raise ValueError("Input and output paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not args.overwrite:
        existing = [
            path
            for path in (output_path, index_path, metadata_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Output already exists; pass --overwrite to replace it: "
                + ", ".join(str(path) for path in existing)
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    temporary_index = index_path.with_name(
        f".{index_path.name}.tmp.{os.getpid()}"
    )
    count = 0

    try:
        with (
            temporary_output.open("wb") as output,
            temporary_index.open("wb") as index,
        ):
            for record_id, record in iter_split_records(
                input_path, args.split, args.backend
            ):
                missing = [
                    field for field in REQUIRED_FIELDS if not record.get(field)
                ]
                if missing:
                    raise KeyError(
                        f"Record {record_id!r} is missing required fields: "
                        + ", ".join(missing)
                    )

                compact_record = {
                    "record_id": str(record_id),
                    "image": record["image"],
                    "target_image": record["target_image"],
                    "ref_image": record["ref_image"],
                    "prompt": record.get("prompt") or args.default_prompt,
                }
                # Online wrapped-reference construction needs the target
                # camera; reference calibration is stored in the Variant-B
                # depth entry. Keep only the optional geometry identity fields
                # that construction needs, not the much larger neighbor list.
                for field in (
                    "camera",
                    "scene",
                    "scene_scale",
                    "geometry_corrupted",
                    "geometry_ref_image",
                    "wrapped_reference_cache_key",
                ):
                    if record.get(field) is not None:
                        compact_record[field] = record[field]
                encoded = (
                    json.dumps(
                        compact_record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
                index.write(struct.pack("<Q", output.tell()))
                output.write(encoded)
                count += 1

                if args.progress_every > 0 and count % args.progress_every == 0:
                    print(f"converted {count:,} records", flush=True)
                if args.limit > 0 and count >= args.limit:
                    break

        os.replace(temporary_output, output_path)
        os.replace(temporary_index, index_path)
        metadata = {
            "schema_version": 1,
            "source": str(input_path),
            "source_size_bytes": input_path.stat().st_size,
            "split": args.split,
            "backend": args.backend,
            "records": count,
            "fields": ["record_id", *REQUIRED_FIELDS, "prompt"],
            "index_format": "little-endian uint64 byte offsets",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        for temporary_path in (temporary_output, temporary_index):
            if temporary_path.exists():
                temporary_path.unlink()

    print(f"wrote {count:,} records to {output_path}")
    print(f"wrote random-access index to {index_path}")


if __name__ == "__main__":
    main(parse_args())
