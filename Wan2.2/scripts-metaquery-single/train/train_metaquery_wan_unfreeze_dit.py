"""
train_metaquery_wan_unfreeze_dit.py
===================================
在不改原始 train_metaquery_wan.py 主流程的前提下，新增一条实验训练范式：

- 训练：Wan DiT + MetaQuery + Connector
- 冻结：MLLM backbone(除可选 MQ embedding)、Wan T5、Wan VAE

说明：
- 该脚本复用原脚本的数据流、loss 和日志框架，仅在“可训练参数组/审计/保存”上做增量改造。
- 默认保存轻量 checkpoint（避免 DiT + optimizer 全量状态过大）。
- 若你确实需要保存全量 DiT/optimizer，可通过环境变量开启（见文末 main 处注释）。
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

import train_metaquery_wan as base_ti2v


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


class MetaQueryWanTrainerUnfreezeDiT(base_ti2v.MetaQueryWanTrainer):
    """
    基于 MetaQueryWanTrainer 的最小侵入扩展：
    1) 解冻并训练 DiT
    2) optimizer 同时覆盖 DiT + MQ/Connector
    3) 审计逻辑改为“期望 DiT 可训练、MLLM/T5/VAE 冻结”
    4) checkpoint 增加 DiT 更新证据（轻量采样快照）
    """

    def __init__(self, args):
        self._dit_param_monitor: List[Any] = []
        self._dit_trainable_param_count = 0
        self._dit_init_trainable_norm = 0.0
        self._dit_init_param_sample_norm = 0.0
        super().__init__(args)

    # ---------------------------------------------------------------------
    # 参数组 & 审计
    # ---------------------------------------------------------------------
    def _dit_learning_rate(self) -> float:
        # 建议 DiT 学习率比 MQ 小，默认 0.1x
        return _env_float("WAN_UNFREEZE_DIT_LR", max(float(self.args.learning_rate) * 0.1, 1e-6))

    def _mq_learning_rate(self) -> float:
        return _env_float("WAN_UNFREEZE_MQ_LR", float(self.args.learning_rate))

    def _dit_weight_decay(self) -> float:
        return _env_float("WAN_UNFREEZE_DIT_WEIGHT_DECAY", 0.01)

    def _mq_weight_decay(self) -> float:
        return _env_float("WAN_UNFREEZE_MQ_WEIGHT_DECAY", 0.1)

    def _dit_trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.wan.model.parameters() if p.requires_grad]

    def _load_models(self):
        super()._load_models()
        # 关键变化：解冻 DiT 并切换 train 模式
        self.wan.model.train().requires_grad_(True)

        # 保持 T5 / VAE 冻结
        try:
            self.wan.text_encoder.model.eval().requires_grad_(False)
        except Exception:
            pass
        try:
            self.wan.vae.model.eval().requires_grad_(False)
        except Exception:
            pass

        if self.is_main_process:
            dit_trainable = sum(int(p.numel()) for p in self.wan.model.parameters() if p.requires_grad)
            print(
                "[UNFREEZE-DIT] Wan DiT 已解冻为可训练 "
                f"(trainable_params={dit_trainable:,})"
            )

    def _setup_optimizer(self):
        args = self.args
        mq_params = self._mq_trainable_params()
        dit_params = self._dit_trainable_params()

        if len(mq_params) == 0:
            raise RuntimeError("[UNFREEZE-DIT] MQ/Connector 可训练参数为空")
        if len(dit_params) == 0:
            raise RuntimeError("[UNFREEZE-DIT] DiT 可训练参数为空，解冻失败")

        mq_count = sum(p.numel() for p in mq_params)
        dit_count = sum(p.numel() for p in dit_params)
        dit_lr = self._dit_learning_rate()
        mq_lr = self._mq_learning_rate()

        print("\n[Optimizer][UNFREEZE-DIT] 可训练参数组:")
        print(f"  DiT                 : {dit_count / 1e6:.1f}M (lr={dit_lr:.2e}, wd={self._dit_weight_decay():.3g})")
        print(f"  MQ+Connector(+emb)  : {mq_count / 1e6:.1f}M (lr={mq_lr:.2e}, wd={self._mq_weight_decay():.3g})")

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": dit_params,
                    "lr": dit_lr,
                    "weight_decay": self._dit_weight_decay(),
                },
                {
                    "params": mq_params,
                    "lr": mq_lr,
                    "weight_decay": self._mq_weight_decay(),
                },
            ],
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(1, args.num_train_steps - args.warmup_steps)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _audit_runtime_trainability(self, stage: str = "runtime", strict: bool | None = None) -> None:
        if strict is None:
            strict = bool(getattr(self.args, "strict_freeze_check", True))

        def stats(module: nn.Module | None) -> Dict[str, int]:
            total, trainable = 0, 0
            if module is None or not isinstance(module, nn.Module):
                return {"total": 0, "trainable": 0}
            for p in module.parameters():
                n = int(p.numel())
                total += n
                if p.requires_grad:
                    trainable += n
            return {"total": total, "trainable": trainable}

        # 期望：DiT可训练，T5/VAE冻结
        dit_stats = stats(getattr(self.wan, "model", None))
        t5_model = getattr(getattr(self.wan, "text_encoder", None), "model", None)
        t5_stats = stats(t5_model)
        vae_model = getattr(getattr(self.wan, "vae", None), "model", None)
        vae_stats = stats(vae_model)

        mq_module = self._mq_encoder_module()
        connector_stats = stats(getattr(getattr(mq_module, "mllm_model", None), "connector", None))
        backbone = getattr(getattr(mq_module, "mllm_model", None), "mllm_backbone", None)
        backbone_stats = stats(backbone)

        emb_trainable = 0
        try:
            emb = backbone.get_input_embeddings()
            if emb is not None and getattr(emb, "weight", None) is not None and emb.weight.requires_grad:
                emb_trainable = int(emb.weight.numel())
        except Exception:
            emb_trainable = 0
        backbone_non_embed_trainable = max(backbone_stats["trainable"] - emb_trainable, 0)

        # optimizer 覆盖关系：应等于 DiT + MQ trainable
        expected_ids = {id(p) for p in (self._dit_trainable_params() + self._mq_trainable_params())}
        opt_ids = []
        for g in self.optimizer.param_groups:
            opt_ids.extend([id(p) for p in g.get("params", [])])
        opt_set = set(opt_ids)
        outside = [x for x in opt_ids if x not in expected_ids]
        missing = [x for x in expected_ids if x not in opt_set]
        dup = max(len(opt_ids) - len(opt_set), 0)

        if self.is_main_process:
            print(
                f"[AUDIT][UNFREEZE-DIT][{stage}] "
                f"dit_trainable={dit_stats['trainable']:,}/{dit_stats['total']:,} "
                f"t5_trainable={t5_stats['trainable']:,}/{t5_stats['total']:,} "
                f"vae_trainable={vae_stats['trainable']:,}/{vae_stats['total']:,} "
                f"connector_trainable={connector_stats['trainable']:,}/{connector_stats['total']:,} "
                f"mllm_backbone_trainable={backbone_stats['trainable']:,} "
                f"(embed_trainable={emb_trainable:,})"
            )
            print(
                f"[AUDIT][UNFREEZE-DIT][{stage}] "
                f"optimizer_params={len(opt_ids)} outside_expected={len(outside)} "
                f"missing_expected={len(missing)} duplicates={dup}"
            )

        errors = []
        if dit_stats["trainable"] <= 0:
            errors.append("DiT 未解冻成功（可训练参数为0）")
        if t5_stats["trainable"] > 0:
            errors.append(f"T5 存在可训练参数: {t5_stats['trainable']}")
        if vae_stats["trainable"] > 0:
            errors.append(f"VAE 存在可训练参数: {vae_stats['trainable']}")
        if connector_stats["trainable"] <= 0:
            errors.append("Connector 可训练参数为0")
        if backbone_non_embed_trainable > 0:
            errors.append(f"MLLM backbone(非embedding)存在可训练参数: {backbone_non_embed_trainable}")
        if (not self.args.train_mq_input_embeddings) and emb_trainable > 0:
            errors.append("设置了冻结 MQ embedding，但 embedding 仍可训练")
        if outside:
            errors.append(f"optimizer 混入非预期参数: {len(outside)}")
        if missing:
            errors.append(f"预期可训练参数未加入 optimizer: {len(missing)}")
        if dup > 0:
            errors.append(f"optimizer 参数重复引用: {dup}")

        if errors:
            msg = " | ".join(errors)
            if strict:
                raise RuntimeError(f"[AUDIT][UNFREEZE-DIT][FAIL][{stage}] {msg}")
            print(f"[AUDIT][UNFREEZE-DIT][WARN][{stage}] {msg}")

    # ---------------------------------------------------------------------
    # 训练期参数变化监控（在原 MQ 监控基础上叠加 DiT）
    # ---------------------------------------------------------------------
    def _init_trainability_monitor(self):
        super()._init_trainability_monitor()
        self._dit_param_monitor = []
        total_sq = 0.0
        sample_sq = 0.0
        total_params = 0
        for name, p in self.wan.model.named_parameters():
            if not p.requires_grad:
                continue
            data = p.detach().float().view(-1)
            numel = int(data.numel())
            if numel <= 0:
                continue
            sample_k = min(8, numel)
            if sample_k == 1:
                idx = torch.zeros(1, dtype=torch.long)
            else:
                idx = torch.linspace(0, numel - 1, steps=sample_k, dtype=torch.long)
            init_vals = data.index_select(0, idx.to(data.device)).cpu()
            self._dit_param_monitor.append((name, p, idx.cpu(), init_vals))
            total_sq += float(torch.sum(data * data).item())
            sample_sq += float(torch.sum(init_vals * init_vals).item())
            total_params += numel
        self._dit_trainable_param_count = total_params
        self._dit_init_trainable_norm = math.sqrt(max(total_sq, 0.0))
        self._dit_init_param_sample_norm = math.sqrt(max(sample_sq, 0.0))
        if self.is_main_process:
            print(
                "[VERIFY][TRAIN-INIT][UNFREEZE-DIT] "
                f"dit_trainable_params={self._dit_trainable_param_count:,} "
                f"dit_init_param_norm={self._dit_init_trainable_norm:.6f} "
                f"dit_monitor_tensors={len(self._dit_param_monitor)}"
            )

    def _collect_trainability_metrics(self):
        metrics = super()._collect_trainability_metrics()
        sample_abs_sum = 0.0
        sample_l2_sum = 0.0
        sample_cur_sq_sum = 0.0
        sample_count = 0
        with torch.no_grad():
            for _, p, idx_cpu, init_vals_cpu in self._dit_param_monitor:
                data = p.detach().float().view(-1)
                idx = idx_cpu.to(data.device)
                now_vals = data.index_select(0, idx).cpu()
                diff = now_vals - init_vals_cpu
                sample_abs_sum += float(diff.abs().sum().item())
                sample_l2_sum += float(torch.sum(diff * diff).item())
                sample_cur_sq_sum += float(torch.sum(now_vals * now_vals).item())
                sample_count += int(diff.numel())
        cur_sample_norm = math.sqrt(max(sample_cur_sq_sum, 0.0))
        init_sample_norm = max(self._dit_init_param_sample_norm, 1e-12)
        metrics.update(
            {
                "train/dit_trainable_param_count": int(self._dit_trainable_param_count),
                "train/dit_param_sample_norm": float(cur_sample_norm),
                "train/dit_param_sample_norm_delta_ratio": float(
                    abs(cur_sample_norm - self._dit_init_param_sample_norm) / init_sample_norm
                ),
                "train/dit_param_sample_abs_delta_mean": float(sample_abs_sum / max(sample_count, 1)),
                "train/dit_param_sample_l2_delta": float(math.sqrt(max(sample_l2_sum, 0.0))),
            }
        )
        return metrics

    def _record_metrics(self, metrics: Dict[str, Any]) -> None:
        super()._record_metrics(metrics)
        if not self._metrics_history:
            return
        row = self._metrics_history[-1]
        for key in (
            "train/dit_trainable_param_count",
            "train/dit_param_sample_norm",
            "train/dit_param_sample_norm_delta_ratio",
            "train/dit_param_sample_abs_delta_mean",
            "train/dit_param_sample_l2_delta",
        ):
            if key in metrics:
                row[key] = metrics[key]

    # ---------------------------------------------------------------------
    # 轻量 checkpoint：默认不保存全量 DiT/optimizer，避免 5B 级文件爆炸
    # ---------------------------------------------------------------------
    def _dit_sample_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        with torch.no_grad():
            for name, p, idx_cpu, _ in self._dit_param_monitor:
                data = p.detach().float().view(-1)
                idx = idx_cpu.to(data.device)
                vals = data.index_select(0, idx).cpu()
                snapshot[name] = {
                    "index": idx_cpu.tolist(),
                    "values": vals.tolist(),
                }
        return snapshot

    def _dit_trainable_stats(self) -> Dict[str, Any]:
        total = 0
        nonzero = 0
        finite_ok = True
        l2_sum = 0.0
        with torch.no_grad():
            for p in self._dit_trainable_params():
                total += int(p.numel())
                nonzero += int((p.detach().abs() > 0).sum().item())
                finite_ok = finite_ok and bool(torch.isfinite(p).all())
                l2_sum += float(p.detach().float().norm().item())
        return {
            "trainable_params": int(total),
            "nonzero_params": int(nonzero),
            "nonzero_ratio": float(nonzero / max(total, 1)),
            "finite_ok": bool(finite_ok),
            "l2_sum": float(l2_sum),
        }

    def _save_checkpoint(self, path, step, extra_info: Dict[str, Any] | None = None):
        if not self.is_main_process:
            return

        path = Path(path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)

        module = self._mq_encoder_module()
        full_state_cpu = base_ti2v._to_cpu_state_dict(module.state_dict())
        name_to_param = dict(module.named_parameters())
        trainable_state_cpu = {
            name: tensor
            for name, tensor in full_state_cpu.items()
            if name_to_param.get(name, None) is not None and name_to_param[name].requires_grad
        }

        # MQ/Connector 主权重（与现有推理脚本兼容）
        torch.save(full_state_cpu, path / "mq_encoder_full.pt")
        torch.save(trainable_state_cpu, path / "mq_encoder_trainable.pt")
        try:
            from safetensors.torch import save_file

            save_file(full_state_cpu, str(path / "model.safetensors"))
            save_file(trainable_state_cpu, str(path / "mq_encoder_trainable.safetensors"))
        except Exception:
            pass

        # 配置与训练元信息
        torch.save(vars(self.args), path / "training_args.bin")
        base_ti2v._write_json(
            path / "training_args.json",
            {str(k): base_ti2v._to_jsonable(v) for k, v in vars(self.args).items()},
        )

        metrics_summary = self._build_metrics_summary(step=step)
        base_ti2v._write_json(path / "metrics_summary.json", metrics_summary)
        base_ti2v._write_json(
            path / "metrics_tail.json",
            {"records": [{str(k): base_ti2v._to_jsonable(v) for k, v in row.items()} for row in self._metrics_history[-200:]]},
        )

        dit_stats = self._dit_trainable_stats()
        base_ti2v._write_json(path / "dit_trainable_stats.json", dit_stats)
        torch.save(self._dit_sample_snapshot(), path / "dit_param_sample.pt")

        trainer_state = {
            "global_step": int(step),
            "checkpoint_format": "wan_metaquery_unfreeze_dit_v1",
            "before_checkpoint_path": self._train_before_checkpoint_path,
            "metrics_jsonl_path": self._metrics_jsonl_path,
            "save_optimizer_state": _env_flag("WAN_UNFREEZE_SAVE_OPTIMIZER", False),
            "save_dit_full": _env_flag("WAN_UNFREEZE_SAVE_DIT_FULL", False),
            "extra_info": base_ti2v._to_jsonable(extra_info or {}),
        }
        base_ti2v._write_json(path / "trainer_state.json", trainer_state)

        # 可选：保存全量 optimizer（非常大）
        if _env_flag("WAN_UNFREEZE_SAVE_OPTIMIZER", False):
            torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
            torch.save(self.scheduler.state_dict(), path / "scheduler.pt")
        else:
            base_ti2v._write_json(
                path / "optimizer_summary.json",
                {
                    "group_count": len(self.optimizer.param_groups),
                    "group_lrs": [float(g.get("lr", 0.0)) for g in self.optimizer.param_groups],
                    "group_param_counts": [int(sum(p.numel() for p in g.get("params", []))) for g in self.optimizer.param_groups],
                },
            )

        # 可选：保存全量 DiT（非常大，默认仅 final）
        save_dit_full = _env_flag("WAN_UNFREEZE_SAVE_DIT_FULL", False)
        save_dit_full_every = _env_flag("WAN_UNFREEZE_SAVE_DIT_FULL_EVERY", False)
        is_final = (path.name == "checkpoint-final")
        if save_dit_full and (save_dit_full_every or is_final):
            print("[UNFREEZE-DIT] 正在保存全量 DiT 权重（可能较慢）...")
            dit_state_cpu = base_ti2v._to_cpu_state_dict(self.wan.model.state_dict())
            torch.save(dit_state_cpu, path / "wan_dit_full.pt")

        # latest 指针
        try:
            with open(path.parent / "latest", "w", encoding="utf-8") as f:
                f.write(f"{path.name}\n")
        except Exception:
            pass

        print(f"  💾 [UNFREEZE-DIT] Checkpoint 已保存: {path}")
        if self.wandb_run is not None and self.args.wandb_log_checkpoint:
            self.wandb.log({"checkpoint/step": int(step), "checkpoint/path": str(path)}, step=step)

    def _wandb_config(self):
        cfg = super()._wandb_config()
        cfg.update(
            {
                "train_mode": "unfreeze_dit",
                "dit_learning_rate": float(self._dit_learning_rate()),
                "mq_learning_rate": float(self._mq_learning_rate()),
                "dit_weight_decay": float(self._dit_weight_decay()),
                "mq_weight_decay": float(self._mq_weight_decay()),
                "save_optimizer_state": bool(_env_flag("WAN_UNFREEZE_SAVE_OPTIMIZER", False)),
                "save_dit_full": bool(_env_flag("WAN_UNFREEZE_SAVE_DIT_FULL", False)),
            }
        )
        return cfg


if __name__ == "__main__":
    # 直接复用原 parse_args（避免改原脚本）
    # 可选环境变量：
    #   WAN_UNFREEZE_DIT_LR=5e-6
    #   WAN_UNFREEZE_MQ_LR=2e-5
    #   WAN_UNFREEZE_SAVE_OPTIMIZER=0
    #   WAN_UNFREEZE_SAVE_DIT_FULL=0
    #   WAN_UNFREEZE_SAVE_DIT_FULL_EVERY=0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise RuntimeError(
            "train_metaquery_wan_unfreeze_dit.py 当前为单进程版本。"
            "若需要多卡同步训练，请先增加 DiT/MQ 的 DDP 封装。"
        )

    args = base_ti2v.parse_args()
    trainer = MetaQueryWanTrainerUnfreezeDiT(args)
    trainer.train()
