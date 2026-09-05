"""Bounded local-video I/O for the native Wan I2V training path."""

from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Type

import numpy as np


def build_timeout_video_dataset_class(base_dataset_class: Type):
    """Add killable ffprobe/ffmpeg timeouts to the upstream Wan dataset.

    OpenCV's ``VideoCapture`` executes in the training process and provides no
    reliable per-file timeout for local/NAS paths.  A wedged read can therefore
    strand one rank while all other ranks wait in an FSDP collective.  This
    adapter performs probing and decoding in subprocesses that can be killed.
    """

    class TimeoutWanVideoDataset(base_dataset_class):
        def __init__(self, *args, **kwargs):
            self.ffprobe_bin = os.environ.get("WAN_DATA_FFPROBE_BIN", "/usr/bin/ffprobe")
            self.ffmpeg_bin = os.environ.get("WAN_DATA_FFMPEG_BIN", "/usr/bin/ffmpeg")
            self.video_probe_timeout_sec = max(
                1.0, float(os.environ.get("WAN_DATA_PROBE_TIMEOUT_SEC", "8"))
            )
            self.video_decode_timeout_sec = max(
                1.0, float(os.environ.get("WAN_DATA_DECODE_TIMEOUT_SEC", "30"))
            )
            self.preclean_cache_wait_timeout_sec = max(
                30.0,
                float(os.environ.get("WAN_DATA_CACHE_WAIT_TIMEOUT_SEC", "1800")),
            )
            self._video_probe_cache = {}
            for executable in (self.ffprobe_bin, self.ffmpeg_bin):
                if not Path(executable).is_file():
                    raise FileNotFoundError(f"bounded video I/O executable missing: {executable}")
            super().__init__(*args, **kwargs)
            # The upstream implementation clamps this to at least 10.  Keep the
            # total worst-case data wait below the 300-second NCCL timeout.
            self.max_trials_cap = max(
                1, int(os.environ.get("WAN_DATA_MAX_TRIALS", "4"))
            )

        def _record_io_failure(self, name: str) -> None:
            stats = getattr(self, "_failure_stats", None)
            if stats is not None:
                stats[name] += 1

        def _run_video_command(self, command, *, timeout: float, failure: str):
            try:
                result = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=float(timeout),
                )
            except subprocess.TimeoutExpired:
                self._record_io_failure(f"{failure}_timeout")
                return None
            except Exception:
                self._record_io_failure(f"{failure}_exception")
                return None
            if result.returncode != 0:
                self._record_io_failure(f"{failure}_failed")
                return None
            return result

        @staticmethod
        def _parse_rate(value) -> float:
            text = str(value or "").strip()
            if not text or text in ("0/0", "N/A"):
                return 0.0
            try:
                if "/" in text:
                    numerator, denominator = text.split("/", 1)
                    denominator_value = float(denominator)
                    return float(numerator) / denominator_value if denominator_value else 0.0
                return float(text)
            except (TypeError, ValueError, ZeroDivisionError):
                return 0.0

        @staticmethod
        def _parse_float(value):
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            return parsed if math.isfinite(parsed) and parsed >= 0.0 else None

        @staticmethod
        def _parse_int(value) -> int:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0

        def _probe_video_metadata(self, video_path):
            key = str(video_path)
            if key in self._video_probe_cache:
                return self._video_probe_cache[key]
            result = self._run_video_command(
                [
                    self.ffprobe_bin,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames,avg_frame_rate,duration:format=duration",
                    "-of",
                    "json",
                    key,
                ],
                timeout=self.video_probe_timeout_sec,
                failure="ffprobe",
            )
            metadata = None
            if result is not None:
                try:
                    payload = json.loads(result.stdout.decode("utf-8", errors="replace"))
                    stream = (payload.get("streams") or [])[0]
                    width = int(stream.get("width", 0) or 0)
                    height = int(stream.get("height", 0) or 0)
                    fps = self._parse_rate(stream.get("avg_frame_rate"))
                    duration = self._parse_float(stream.get("duration"))
                    if duration is None:
                        duration = self._parse_float((payload.get("format") or {}).get("duration"))
                    frame_count = self._parse_int(stream.get("nb_frames"))
                    if frame_count <= 0 and duration is not None and fps > 0:
                        frame_count = int(duration * fps + 1e-6)
                    if width > 0 and height > 0:
                        metadata = {
                            "width": width,
                            "height": height,
                            "fps": fps,
                            "duration": duration,
                            "frame_count": frame_count,
                        }
                except Exception:
                    self._record_io_failure("ffprobe_parse_failed")
            self._video_probe_cache[key] = metadata
            return metadata

        def _metadata_meets_contract(self, metadata) -> bool:
            if metadata is None:
                return False
            frame_count = int(metadata.get("frame_count", 0))
            if frame_count > 0 and frame_count < int(self.frame_num):
                return False
            duration = metadata.get("duration")
            if duration is not None and duration < float(self.min_duration_sec):
                return False
            if duration is not None and duration > float(self.max_duration_sec):
                return False
            return True

        def _resolve_openvid_local_path(self, value: str):
            # Local subset records already contain an absolute path.  Returning
            # it directly avoids an unbounded Path.exists() on a stalled NAS.
            direct = Path(str(value))
            if direct.is_absolute():
                return direct
            return super()._resolve_openvid_local_path(value)

        def _is_cache_file_valid(self, file_path: Path):
            return self._metadata_meets_contract(
                self._probe_video_metadata(str(file_path))
            )

        def _probe_video_quick(self, video_path):
            metadata = self._probe_video_metadata(video_path)
            if not self._metadata_meets_contract(metadata):
                return False
            result = self._run_video_command(
                [
                    self.ffmpeg_bin,
                    "-nostdin",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    str(video_path),
                    "-map",
                    "0:v:0",
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ],
                timeout=self.video_probe_timeout_sec,
                failure="ffmpeg_probe",
            )
            return result is not None

        def _load_video_frames(self, video_path):
            metadata = self._probe_video_metadata(video_path)
            if not self._metadata_meets_contract(metadata):
                return None, None, None
            height = int(metadata["height"])
            width = int(metadata["width"])
            area = height * width
            if area > int(self.max_area):
                scale = math.sqrt(float(self.max_area) / area)
                height = int(height * scale)
                width = int(width * scale)
            height = max(32, (height // 32) * 32)
            width = max(32, (width // 32) * 32)
            result = self._run_video_command(
                [
                    self.ffmpeg_bin,
                    "-nostdin",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    str(video_path),
                    "-map",
                    "0:v:0",
                    "-vf",
                    f"scale={width}:{height}:flags=area",
                    "-frames:v",
                    str(self.frame_num),
                    "-an",
                    "-sn",
                    "-dn",
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ],
                timeout=self.video_decode_timeout_sec,
                failure="ffmpeg_decode",
            )
            if result is None:
                return None, None, None
            expected = int(self.frame_num) * height * width * 3
            if len(result.stdout) != expected:
                self._record_io_failure("ffmpeg_decode_short")
                return None, None, None
            array = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
                int(self.frame_num), height, width, 3
            )
            frames = [array[index] for index in range(int(self.frame_num))]
            return frames, float(metadata.get("fps", 0.0)), metadata.get("duration")

        def _build_local_openvid_subset(self, cache_path):
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if not self.preclean_enabled or world_size <= 1 or rank == 0:
                return super()._build_local_openvid_subset(cache_path)

            # Only rank 0 scans/decodes the NAS pool.  Other ranks wait on the
            # small pickle cache without entering a distributed collective, so
            # the 300-second NCCL timeout is not consumed by preprocessing.
            deadline = time.monotonic() + self.preclean_cache_wait_timeout_sec
            last_error = None
            while time.monotonic() < deadline:
                try:
                    with open(cache_path, "rb") as handle:
                        payload = pickle.load(handle)
                    samples = payload.get("samples", [])
                    if samples:
                        self._subset_scanned = int(payload.get("scanned", 0))
                        self._subset_accepted = len(samples)
                        rejected = payload.get("rejected_stats", {})
                        self._subset_rejected_stats.update(
                            {str(key): int(value) for key, value in rejected.items()}
                        )
                        print(
                            f"[WanVideoDataset] rank={rank} bounded_preclean_cache_hit="
                            f"{cache_path} loaded={len(samples)}",
                            flush=True,
                        )
                        return samples
                except (FileNotFoundError, EOFError, pickle.UnpicklingError) as error:
                    last_error = error
                except Exception as error:
                    last_error = error
                time.sleep(0.5)
            raise TimeoutError(
                f"rank {rank} timed out waiting for rank-0 preclean cache {cache_path}; "
                f"last_error={last_error!r}"
            )

    TimeoutWanVideoDataset.__name__ = "TimeoutWanVideoDataset"
    return TimeoutWanVideoDataset
