#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


VIDEO_COLUMNS = ("video", "video_path", "filename", "file", "path")
CAPTION_COLUMNS = ("caption", "text", "description", "prompt", "summary")
MAX_MISSING_EXAMPLES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, symlink-only OpenVid subset in CSV order."
    )
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--probe_training_video", action="store_true")
    parser.add_argument("--frame_num", type=int, default=49)
    parser.add_argument("--min_duration_sec", type=float, default=0.5)
    parser.add_argument("--max_duration_sec", type=float, default=20.0)
    parser.add_argument("--max_caption_tokens", type=int, default=512)
    parser.add_argument("--caption_tokenizer_path", type=Path, default=None)
    parser.add_argument("--candidate_manifest", type=Path, default=None)
    parser.add_argument("--probe_workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_columns(fieldnames: Iterable[str]) -> Tuple[str, str]:
    lowered = {str(name).strip().lower(): str(name) for name in fieldnames}
    video = next((lowered[name] for name in VIDEO_COLUMNS if name in lowered), None)
    caption = next(
        (lowered[name] for name in CAPTION_COLUMNS if name in lowered), None
    )
    if video is None or caption is None:
        raise ValueError(
            f"CSV requires video/caption columns; available={list(fieldnames)}"
        )
    return video, caption


def resolve_video(video_root: Path, raw_value: str) -> Optional[Path]:
    value = raw_value.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    candidates = [video_root / value.lstrip("/"), video_root / Path(value).name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def probe_training_video(
    video_path: Path,
    *,
    frame_num: int,
    min_duration_sec: float,
    max_duration_sec: float,
) -> Tuple[bool, str]:
    """Apply the same basic video contract used by WanVideoDataset."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return False, "open_failed"
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if total < int(frame_num):
            return False, "too_few_frames"
        duration = (total / fps) if fps > 0.0 else None
        if duration is not None and duration < float(min_duration_sec):
            return False, "duration_too_short"
        if duration is not None and duration > float(max_duration_sec):
            return False, "duration_too_long"
        # Match WanVideoDataset's quick pre-clean.  Whole-video random reference
        # decoding has its own deterministic eight-position retry at read time;
        # seeking three times here makes a 4000-video launch prohibitively slow.
        ok, frame = capture.read()
        if not ok or frame is None:
            return False, "first_frame_decode_failed"
        return True, "ok"
    finally:
        capture.release()


def print_report_summary(report: Dict[str, object]) -> None:
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"records", "missing_before_limit_examples"}
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.frame_num <= 0:
        raise ValueError("--frame_num must be positive")
    if not 0.0 <= args.min_duration_sec <= args.max_duration_sec:
        raise ValueError("invalid duration range")
    if args.max_caption_tokens <= 0:
        raise ValueError("--max_caption_tokens must be positive")
    if args.probe_workers <= 0:
        raise ValueError("--probe_workers must be positive")
    video_root = args.video_root.expanduser().resolve()
    csv_path = args.csv_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    subset_video_root = output_root / "videos"
    subset_csv = output_root / f"openvid_first{args.limit}.csv"
    report_path = output_root / "manifest.json"

    if not video_root.is_dir():
        raise FileNotFoundError(f"video root not found: {video_root}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if output_root.exists() and not args.overwrite:
        if subset_csv.is_file() and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            expected_probe = {
                "enabled": bool(args.probe_training_video),
                "frame_num": int(args.frame_num),
                "min_duration_sec": float(args.min_duration_sec),
                "max_duration_sec": float(args.max_duration_sec),
                "max_caption_tokens": int(args.max_caption_tokens),
                "caption_tokenizer_path": (
                    str(args.caption_tokenizer_path.expanduser().resolve())
                    if args.caption_tokenizer_path is not None
                    else None
                ),
            }
            if (
                int(report.get("kept_count", 0)) == args.limit
                and report.get("training_probe") == expected_probe
            ):
                print_report_summary(report)
                return
        raise FileExistsError(
            f"output exists but is incomplete: {output_root}; pass --overwrite"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    subset_video_root.mkdir(parents=True, exist_ok=True)

    kept = []
    missing_count = 0
    missing_examples = []
    rejected_counts: Dict[str, int] = {}
    seen_targets = set()
    tokenizer = None
    tokenizer_path = None
    seed_records: Dict[int, Dict[str, object]] = {}
    max_seed_row = -1
    if args.candidate_manifest is not None:
        candidate_manifest = args.candidate_manifest.expanduser().resolve()
        if not candidate_manifest.is_file():
            raise FileNotFoundError(
                f"candidate manifest not found: {candidate_manifest}"
            )
        candidate_payload = json.loads(
            candidate_manifest.read_text(encoding="utf-8")
        )
        for record in candidate_payload.get("records", []):
            row_index = int(record["row"])
            seed_records[row_index] = dict(record)
            max_seed_row = max(max_seed_row, row_index)
    if args.probe_training_video and args.caption_tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer_path = args.caption_tokenizer_path.expanduser().resolve()
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"caption tokenizer not found: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
        )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        video_col, caption_col = find_columns(reader.fieldnames or [])
        enumerated_rows = iter(enumerate(reader))
        probe = partial(
            probe_training_video,
            frame_num=args.frame_num,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
        )
        executor = (
            ProcessPoolExecutor(max_workers=int(args.probe_workers))
            if args.probe_training_video
            else None
        )
        exhausted = False
        try:
            while len(kept) < args.limit and not exhausted:
                candidates = []
                batch_target = max(1, int(args.probe_workers) * 4)
                while len(candidates) < batch_target:
                    try:
                        row_index, row = next(enumerated_rows)
                    except StopIteration:
                        exhausted = True
                        break
                    raw_video = str(row.get(video_col, "") or "").strip()
                    caption = str(row.get(caption_col, "") or "").strip()
                    if row_index <= max_seed_row:
                        seed_record = seed_records.get(row_index)
                        if seed_record is None:
                            continue
                        caption = str(
                            seed_record.get("caption", "") or ""
                        ).strip()
                        source_video = Path(
                            str(seed_record["source"])
                        ).expanduser()
                        raw_video = str(
                            seed_record.get("video", source_video.name)
                        )
                    else:
                        source_video = None
                    if not raw_video or not caption:
                        continue
                    if source_video is None:
                        source_video = resolve_video(video_root, raw_video)
                    if source_video is None:
                        missing_count += 1
                        if len(missing_examples) < MAX_MISSING_EXAMPLES:
                            missing_examples.append(
                                {"row": row_index, "video": raw_video}
                            )
                        continue
                    if tokenizer is not None:
                        token_count = len(
                            tokenizer(
                                caption,
                                add_special_tokens=True,
                                truncation=False,
                            )["input_ids"]
                        )
                        if token_count > args.max_caption_tokens:
                            reason = "caption_too_long"
                            rejected_counts[reason] = (
                                rejected_counts.get(reason, 0) + 1
                            )
                            continue
                    candidates.append((row_index, caption, source_video))

                if not candidates:
                    continue
                if executor is None:
                    probe_results = [(True, "ok")] * len(candidates)
                else:
                    probe_results = list(
                        executor.map(probe, [row[2] for row in candidates])
                    )

                for (row_index, caption, source_video), (
                    accepted,
                    reason,
                ) in zip(candidates, probe_results):
                    if not accepted:
                        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                        continue
                    target_name = source_video.name
                    if target_name in seen_targets:
                        target_name = f"{row_index:08d}_{target_name}"
                    seen_targets.add(target_name)
                    target = subset_video_root / target_name
                    if target.exists() or target.is_symlink():
                        if args.overwrite:
                            target.unlink()
                        else:
                            raise FileExistsError(target)
                    os.symlink(source_video, target)
                    kept.append(
                        {
                            "row": row_index,
                            "video": target_name,
                            "caption": caption,
                            "source": str(source_video),
                        }
                    )
                    if len(kept) >= args.limit:
                        break
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    if len(kept) != args.limit:
        raise RuntimeError(
            f"only resolved {len(kept)} training-valid records, expected "
            f"{args.limit}; missing={missing_count}, rejected={rejected_counts}"
        )

    if args.overwrite:
        selected_names = {str(record["video"]) for record in kept}
        for existing in subset_video_root.iterdir():
            if existing.name not in selected_names and existing.is_symlink():
                existing.unlink()

    with subset_csv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["video", "caption"])
        writer.writeheader()
        writer.writerows(
            {"video": row["video"], "caption": row["caption"]} for row in kept
        )

    report: Dict[str, object] = {
        "format": "moviestory_openvid_subset_v2",
        "selection": "CSV order, first N training-valid records",
        "requested_limit": args.limit,
        "kept_count": len(kept),
        "source_video_root": str(video_root),
        "source_csv": str(csv_path),
        "subset_video_root": str(subset_video_root),
        "subset_csv": str(subset_csv),
        "missing_before_limit_count": missing_count,
        "missing_before_limit_examples": missing_examples,
        "missing_examples_truncated": missing_count > len(missing_examples),
        "rejected_before_limit": rejected_counts,
        "training_probe": {
            "enabled": bool(args.probe_training_video),
            "frame_num": int(args.frame_num),
            "min_duration_sec": float(args.min_duration_sec),
            "max_duration_sec": float(args.max_duration_sec),
            "max_caption_tokens": int(args.max_caption_tokens),
            "caption_tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        },
        "records": kept,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_report_summary(report)


if __name__ == "__main__":
    main()
