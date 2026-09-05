#!/usr/bin/env bash

# Populate WANDB_ARGS without ever placing WANDB_API_KEY on the command line.
configure_wandb_args() {
  local default_run_name="$1"
  local default_tags="$2"
  local enabled="${WANDB_ENABLED:-auto}"
  local mode="${WANDB_MODE:-online}"

  WANDB_ARGS=()

  case "${enabled,,}" in
    auto)
      if [[ -n "${WANDB_API_KEY:-}" || "${mode}" == "offline" ]]; then
        enabled=1
      else
        enabled=0
      fi
      ;;
    1|true|yes|on) enabled=1 ;;
    0|false|no|off) enabled=0 ;;
    *)
      echo "Invalid WANDB_ENABLED=${enabled}; use auto, 1, or 0" >&2
      return 2
      ;;
  esac

  if [[ "${enabled}" == "0" ]]; then
    echo "[W&B] disabled (export WANDB_API_KEY or set WANDB_ENABLED=1)"
    return
  fi

  case "${mode}" in
    online|offline|disabled) ;;
    *)
      echo "Invalid WANDB_MODE=${mode}; use online, offline, or disabled" >&2
      return 2
      ;;
  esac

  if [[ "${mode}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "[W&B] WANDB_API_KEY is not exported; using credentials from wandb login"
  fi

  WANDB_ARGS+=(
    --wandb_enabled
    --wandb_project "${WANDB_PROJECT:-moviestory-3router}"
    --wandb_run_name "${WANDB_RUN_NAME:-${default_run_name}}"
    --wandb_tags "${WANDB_TAGS:-${default_tags}}"
    --wandb_mode "${mode}"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi

  case "${WANDB_LOG_EVERY_STEP:-1}" in
    1|true|yes|on) WANDB_ARGS+=(--wandb_log_every_step) ;;
    0|false|no|off) ;;
    *)
      echo "Invalid WANDB_LOG_EVERY_STEP=${WANDB_LOG_EVERY_STEP}" >&2
      return 2
      ;;
  esac
  case "${WANDB_LOG_CHECKPOINT:-0}" in
    1|true|yes|on) WANDB_ARGS+=(--wandb_log_checkpoint) ;;
    0|false|no|off) ;;
    *)
      echo "Invalid WANDB_LOG_CHECKPOINT=${WANDB_LOG_CHECKPOINT}" >&2
      return 2
      ;;
  esac
}
