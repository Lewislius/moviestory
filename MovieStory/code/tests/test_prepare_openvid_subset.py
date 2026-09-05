import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PrepareOpenVidSubsetTest(unittest.TestCase):
    def test_selects_first_resolvable_rows_in_csv_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "source_videos"
            videos.mkdir()
            csv_path = root / "source.csv"
            output = root / "subset"

            rows = []
            for index in range(105):
                name = f"clip_{index:03d}.mp4"
                (videos / name).touch()
                rows.append({"video": name, "caption": f"caption {index}"})
            rows.insert(3, {"video": "missing.mp4", "caption": "missing"})
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video", "caption"])
                writer.writeheader()
                writer.writerows(rows)

            script = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "prepare_openvid_subset.py"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--video_root",
                    str(videos),
                    "--csv_path",
                    str(csv_path),
                    "--output_root",
                    str(output),
                    "--limit",
                    "100",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["kept_count"], 100)
            self.assertEqual(manifest["records"][0]["video"], "clip_000.mp4")
            self.assertEqual(manifest["records"][-1]["video"], "clip_099.mp4")
            self.assertEqual(manifest["missing_before_limit_count"], 1)
            self.assertEqual(len(manifest["missing_before_limit_examples"]), 1)
            self.assertFalse(manifest["missing_examples_truncated"])
            self.assertTrue(
                (output / "videos" / "clip_000.mp4").is_symlink()
            )

    def test_candidate_manifest_is_reused_then_topped_up_in_csv_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "source_videos"
            videos.mkdir()
            csv_path = root / "source.csv"
            output = root / "subset"
            candidate_manifest = root / "candidate.json"
            rows = []
            records = []
            for index in range(6):
                name = f"clip_{index:03d}.mp4"
                source_video = videos / name
                source_video.touch()
                caption = f"caption {index}"
                rows.append({"video": name, "caption": caption})
                if index < 3:
                    records.append(
                        {
                            "row": index,
                            "video": name,
                            "caption": caption,
                            "source": str(source_video.resolve()),
                        }
                    )
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video", "caption"])
                writer.writeheader()
                writer.writerows(rows)
            candidate_manifest.write_text(
                json.dumps({"records": records}),
                encoding="utf-8",
            )
            script = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "prepare_openvid_subset.py"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--video_root",
                    str(videos),
                    "--csv_path",
                    str(csv_path),
                    "--output_root",
                    str(output),
                    "--limit",
                    "5",
                    "--candidate_manifest",
                    str(candidate_manifest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [record["row"] for record in manifest["records"]],
                [0, 1, 2, 3, 4],
            )


if __name__ == "__main__":
    unittest.main()
