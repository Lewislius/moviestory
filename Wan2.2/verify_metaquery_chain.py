#!/usr/bin/env python
"""
verify_metaquery_chain.py
=========================
训练/推理链路审计工具：
1) 对比训练前后 checkpoint 参数变化（重点检查 connector 是否更新）
2) 汇总 inference_metaquery_wan.py 产出的 *.verify.json 报告
3) 输出统一诊断结论，帮助定位训练链路 or 推理链路问题
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict

import torch


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        tensor_values = [v for v in payload.values() if torch.is_tensor(v)]
        non_tensor_values = [v for v in payload.values() if not torch.is_tensor(v)]
        if tensor_values and not non_tensor_values:
            return payload
    raise ValueError("无法从 checkpoint payload 中提取 state_dict")


def resolve_checkpoint_file(path_or_dir: str) -> Path:
    path = Path(path_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 路径不存在: {path}")
    if path.is_file():
        return path
    candidates = [
        path / "mq_encoder_full.pt",
        path / "mq_encoder_full.safetensors",
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "training_state.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"checkpoint 目录中未找到可加载权重: {path}，"
        f"候选={[c.name for c in candidates]}"
    )


def load_state_dict(path_or_dir: str) -> tuple[Dict[str, torch.Tensor], Path]:
    file_path = resolve_checkpoint_file(path_or_dir)
    if file_path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(f"加载 safetensors 失败: {file_path}") from e
        state = load_file(str(file_path), device="cpu")
    else:
        payload = safe_torch_load(file_path, map_location="cpu")
        state = extract_state_dict(payload)
    return state, file_path


def accumulate_delta_stats(before: torch.Tensor, after: torch.Tensor, eps: float) -> Dict[str, float]:
    b = before.detach().to("cpu").reshape(-1)
    a = after.detach().to("cpu").reshape(-1)
    n = b.numel()
    chunk = 1_000_000
    changed = 0
    max_abs = 0.0
    sum_abs = 0.0
    for i in range(0, n, chunk):
        bb = b[i:i + chunk].to(torch.float32)
        aa = a[i:i + chunk].to(torch.float32)
        d = (aa - bb).abs()
        changed += int((d > eps).sum().item())
        if d.numel() > 0:
            local_max = float(d.max().item())
            if local_max > max_abs:
                max_abs = local_max
            sum_abs += float(d.sum().item())
    return {
        "numel": int(n),
        "changed_elems": int(changed),
        "changed_ratio": float(changed / max(n, 1)),
        "max_abs_delta": float(max_abs),
        "mean_abs_delta": float(sum_abs / max(n, 1)),
    }


def summarize_group(
    before_state: Dict[str, torch.Tensor],
    after_state: Dict[str, torch.Tensor],
    keys: list[str],
    eps: float,
) -> Dict[str, Any]:
    out = {
        "tensor_count": 0,
        "numel": 0,
        "changed_elems": 0,
        "changed_ratio": 0.0,
        "max_abs_delta": 0.0,
        "mean_abs_delta_weighted": 0.0,
        "shape_mismatch_tensors": 0,
    }
    weighted_sum = 0.0
    valid = 0
    for key in keys:
        b = before_state[key]
        a = after_state[key]
        if b.shape != a.shape:
            out["shape_mismatch_tensors"] += 1
            continue
        stats = accumulate_delta_stats(b, a, eps=eps)
        out["numel"] += stats["numel"]
        out["changed_elems"] += stats["changed_elems"]
        if stats["max_abs_delta"] > out["max_abs_delta"]:
            out["max_abs_delta"] = stats["max_abs_delta"]
        weighted_sum += stats["mean_abs_delta"] * stats["numel"]
        valid += 1
    out["tensor_count"] = valid
    out["changed_ratio"] = float(out["changed_elems"] / max(out["numel"], 1))
    out["mean_abs_delta_weighted"] = float(weighted_sum / max(out["numel"], 1))
    return out


def read_trainer_state(ckpt_file: Path) -> Dict[str, Any]:
    ckpt_dir = ckpt_file.parent
    trainer_state_file = ckpt_dir / "trainer_state.json"
    if not trainer_state_file.exists():
        return {"exists": False}
    try:
        data = json.loads(trainer_state_file.read_text(encoding="utf-8"))
        return {
            "exists": True,
            "global_step": int(data.get("global_step", 0)),
            "checkpoint_format": str(data.get("checkpoint_format", "")),
        }
    except Exception as e:
        return {
            "exists": True,
            "parse_error": str(e),
        }


def read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"exists": True, "data": data}
    except Exception as e:
        return {"exists": True, "parse_error": str(e)}


def inspect_artifacts(ckpt_file: Path) -> Dict[str, Any]:
    ckpt_dir = ckpt_file.parent
    return {
        "checkpoint_dir": str(ckpt_dir),
        "config.json": (ckpt_dir / "config.json").exists(),
        "trainer_state.json": (ckpt_dir / "trainer_state.json").exists(),
        "optimizer.pt": (ckpt_dir / "optimizer.pt").exists(),
        "scheduler.pt": (ckpt_dir / "scheduler.pt").exists(),
        "training_args.bin": (ckpt_dir / "training_args.bin").exists(),
        "training_args.json": (ckpt_dir / "training_args.json").exists(),
        "mq_encoder_trainable.pt": (ckpt_dir / "mq_encoder_trainable.pt").exists(),
        "mq_encoder_trainable.safetensors": (ckpt_dir / "mq_encoder_trainable.safetensors").exists(),
        "model.safetensors": (ckpt_dir / "model.safetensors").exists(),
        "mq_encoder_full.pt": (ckpt_dir / "mq_encoder_full.pt").exists(),
        "metrics_summary.json": (ckpt_dir / "metrics_summary.json").exists(),
        "metrics_tail.json": (ckpt_dir / "metrics_tail.json").exists(),
    }


def analyze_inference_reports(glob_pattern: str) -> Dict[str, Any]:
    paths = sorted(glob.glob(glob_pattern))
    result: Dict[str, Any] = {
        "glob_pattern": glob_pattern,
        "report_count": len(paths),
        "reports": [],
        "warnings": [],
        "failures": [],
    }
    for p in paths:
        path = Path(p)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            result["failures"].append(f"{p}: 读取失败: {e}")
            continue
        runtime = data.get("runtime", {})
        summary = data.get("summary", {})
        report_warnings = data.get("warnings", [])
        item = {
            "path": str(path),
            "summary_status": summary.get("status", ""),
            "warning_count": len(report_warnings),
            "mq_influence_keys": {},
        }
        for k, v in runtime.items():
            if k.endswith("_mq_influence_diff_norm") or k.endswith("_mq_cond_uncond_diff_norm"):
                item["mq_influence_keys"][k] = float(v)
        for k, v in item["mq_influence_keys"].items():
            if k.endswith("_mq_influence_diff_norm") and float(v) <= 1e-6:
                result["failures"].append(f"{path.name}: {k}={v} (MQ 对 Wan 影响过低)")
            if k.endswith("_mq_cond_uncond_diff_norm") and float(v) <= 1e-6:
                result["warnings"].append(f"{path.name}: {k}={v} (MQ cond/uncond 差异过低)")
        if report_warnings:
            result["warnings"].append(f"{path.name}: 内部 warnings={len(report_warnings)}")
        result["reports"].append(item)
    return result


def compare_checkpoints(before_ckpt: str, after_ckpt: str, eps: float) -> Dict[str, Any]:
    before_state, before_file = load_state_dict(before_ckpt)
    after_state, after_file = load_state_dict(after_ckpt)

    shared_keys = [k for k in after_state.keys() if k in before_state]
    connector_keys = [k for k in shared_keys if "mllm_model.connector" in k]
    embed_keys = [k for k in shared_keys if "embed_tokens.weight" in k]
    backbone_other_keys = [
        k for k in shared_keys
        if "mllm_model.mllm_backbone" in k and "embed_tokens.weight" not in k
    ]

    all_summary = summarize_group(before_state, after_state, shared_keys, eps)
    connector_summary = summarize_group(before_state, after_state, connector_keys, eps)
    embed_summary = summarize_group(before_state, after_state, embed_keys, eps)
    backbone_other_summary = summarize_group(before_state, after_state, backbone_other_keys, eps)

    warnings = []
    failures = []

    if len(shared_keys) == 0:
        failures.append("before/after checkpoint 没有共享参数键，无法比较")
    if connector_summary["tensor_count"] == 0:
        failures.append("未发现 connector 参数键，checkpoint 类型可能不匹配")
    elif connector_summary["changed_elems"] == 0:
        failures.append("connector 参数未变化，训练可能未生效或路径指错")
    if all_summary["changed_elems"] == 0:
        failures.append("全部参数未变化，before 与 after 实际相同")
    if backbone_other_summary["changed_ratio"] > 1e-5:
        warnings.append(
            f"backbone(非embed) 参数变化比例偏高: {backbone_other_summary['changed_ratio']:.3e}，"
            "若你期望其冻结，需要复查训练配置"
        )

    before_ts = read_trainer_state(before_file)
    after_ts = read_trainer_state(after_file)
    before_args = read_optional_json(before_file.parent / "training_args.json")
    after_args = read_optional_json(after_file.parent / "training_args.json")
    before_metrics = read_optional_json(before_file.parent / "metrics_summary.json")
    after_metrics = read_optional_json(after_file.parent / "metrics_summary.json")
    chain_manifest = read_optional_json(after_file.parent.parent / "training_chain_manifest.json")
    if before_ts.get("exists") and after_ts.get("exists"):
        b = int(before_ts.get("global_step", 0))
        a = int(after_ts.get("global_step", 0))
        if a <= b:
            warnings.append(f"trainer_state.global_step 未增加: before={b}, after={a}")

    return {
        "before_checkpoint_input": before_ckpt,
        "after_checkpoint_input": after_ckpt,
        "before_checkpoint_resolved": str(before_file),
        "after_checkpoint_resolved": str(after_file),
        "before_artifacts": inspect_artifacts(before_file),
        "after_artifacts": inspect_artifacts(after_file),
        "before_trainer_state": before_ts,
        "after_trainer_state": after_ts,
        "before_training_args_json": before_args,
        "after_training_args_json": after_args,
        "before_metrics_summary_json": before_metrics,
        "after_metrics_summary_json": after_metrics,
        "training_chain_manifest_json": chain_manifest,
        "before_key_count": len(before_state),
        "after_key_count": len(after_state),
        "shared_key_count": len(shared_keys),
        "all": all_summary,
        "connector": connector_summary,
        "embed_tokens": embed_summary,
        "backbone_other": backbone_other_summary,
        "warnings": warnings,
        "failures": failures,
    }


def parse_args():
    p = argparse.ArgumentParser(description="MetaQuery 训练/推理链路审计")
    p.add_argument("--before_checkpoint", type=str, required=True, help="训练前 checkpoint（基线）")
    p.add_argument("--after_checkpoint", type=str, required=True, help="训练后 checkpoint（当前推理使用）")
    p.add_argument("--inference_report_glob", type=str, default="", help="推理 verify json 的 glob，例如 outputs/*.verify.json")
    p.add_argument("--eps", type=float, default=1e-7, help="参数变化判定阈值")
    p.add_argument("--output_json", type=str, default="", help="输出报告 json 路径")
    p.add_argument("--strict", action="store_true", help="审计失败时返回非零退出码")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_cmp = compare_checkpoints(
        before_ckpt=args.before_checkpoint,
        after_ckpt=args.after_checkpoint,
        eps=args.eps,
    )

    inf_result = None
    if args.inference_report_glob:
        inf_result = analyze_inference_reports(args.inference_report_glob)

    training_ok = len(ckpt_cmp["failures"]) == 0
    inference_ok = None
    if inf_result is not None:
        inference_ok = len(inf_result["failures"]) == 0

    overall_ok = training_ok and (inference_ok if inference_ok is not None else True)
    report = {
        "training_checkpoint_compare": ckpt_cmp,
        "inference_reports_analysis": inf_result,
        "verdict": {
            "training_chain_ok": training_ok,
            "inference_chain_ok": inference_ok,
            "overall_ok": overall_ok,
        },
    }

    print("=" * 70)
    print("[CHAIN-AUDIT] 训练前后 checkpoint 比较")
    print(f"  shared_key_count            : {ckpt_cmp['shared_key_count']}")
    print(f"  connector.changed_ratio     : {ckpt_cmp['connector']['changed_ratio']:.6e}")
    print(f"  connector.max_abs_delta     : {ckpt_cmp['connector']['max_abs_delta']:.6e}")
    print(f"  all.changed_ratio           : {ckpt_cmp['all']['changed_ratio']:.6e}")
    after_metrics = ckpt_cmp.get("after_metrics_summary_json", {})
    if after_metrics.get("exists") and isinstance(after_metrics.get("data"), dict):
        ms = after_metrics["data"]
        print(f"  metrics.logged_steps        : {ms.get('logged_steps', 'n/a')}")
        print(f"  metrics.loss_last           : {ms.get('loss_last', 'n/a')}")
        print(f"  metrics.lr_last             : {ms.get('lr_last', 'n/a')}")
    print(f"  training_chain_ok           : {training_ok}")
    if ckpt_cmp["warnings"]:
        print("  warnings:")
        for w in ckpt_cmp["warnings"]:
            print(f"    - {w}")
    if ckpt_cmp["failures"]:
        print("  failures:")
        for f in ckpt_cmp["failures"]:
            print(f"    - {f}")

    if inf_result is not None:
        print("[CHAIN-AUDIT] 推理验证报告汇总")
        print(f"  report_count                : {inf_result['report_count']}")
        print(f"  inference_chain_ok          : {inference_ok}")
        if inf_result["warnings"]:
            print("  warnings:")
            for w in inf_result["warnings"]:
                print(f"    - {w}")
        if inf_result["failures"]:
            print("  failures:")
            for f in inf_result["failures"]:
                print(f"    - {f}")

    print(f"[CHAIN-AUDIT] overall_ok={overall_ok}")
    print("=" * 70)

    output_path = args.output_json.strip()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[CHAIN-AUDIT] 报告已写入: {out}")

    if args.strict and (not overall_ok):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
