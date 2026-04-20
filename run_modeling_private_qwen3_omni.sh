#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENVINO_BIN="${ROOT_DIR}/openvino/bin/intel64/Release"
GENAI_DLL_DIR="${ROOT_DIR}/openvino.genai/build/openvino_genai"
GENAI_RUNTIME_BIN_DIR="${ROOT_DIR}/openvino.genai/build/bin"
GENAI_RELEASE_BIN_DIR="${GENAI_RUNTIME_BIN_DIR}/Release"
BRIDGE_DIR="${ROOT_DIR}/modeling_private/models/qwen3_omni"
VENV_BIN="${ROOT_DIR}/.venv/bin"

log_error() {
    echo "[ERROR] $*" >&2
}

log_warn() {
    echo "[WARN] $*" >&2
}

usage() {
    cat <<'EOF'
Usage:
  run_modeling_private_qwen3_omni.sh image [--repeat N] --model-dir PATH --image PATH [--prompt TEXT] [--device NAME] [--output-tokens N] [--cache-model] [--vision-quant-mode MODE] [--vision-group-size N] [--vision-backup-mode MODE] [--text-quant-mode MODE] [--text-group-size N] [--text-backup-mode MODE]
  run_modeling_private_qwen3_omni.sh tts [--repeat N] MODEL_DIR CASE_ID TEXT_PROMPT WAV_OUT [IMAGE_PATH] [AUDIO_PATH] [DEVICE] [MAX_NEW_TOKENS] [PRECISION] [VIDEO_FRAMES_DIR] [--cache-model]

Examples:
  bash run_modeling_private_qwen3_omni.sh image --repeat 5 --model-dir /data/models/Huggingface/Qwen3-Omni --image /data/images/cat.jpg --prompt "What can you see" --device CPU --cache-model --vision-quant-mode int8_asym --vision-group-size 128 --text-quant-mode int4_asym --text-group-size 128 --text-backup-mode int8_asym
  bash run_modeling_private_qwen3_omni.sh tts /data/models/Huggingface/Qwen3-Omni demo "Describe this scene" /tmp/omni.wav /data/images/cat.jpg "" CPU 64 fp32 --cache-model
EOF
}

find_tbb_dir() {
    local candidate

    shopt -s nullglob
    for candidate in \
        "${ROOT_DIR}"/openvino/temp/*/tbb/bin \
        "${ROOT_DIR}"/openvino/temp/*/tbb/lib \
        "${ROOT_DIR}"/openvino/temp/*/tbb/lib64; do
        if [[ -d "${candidate}" ]] && compgen -G "${candidate}/libtbb.so*" >/dev/null; then
            printf '%s\n' "${candidate}"
            shopt -u nullglob
            return 0
        fi
    done
    shopt -u nullglob
    return 1
}

prepend_env_path() {
    local var_name="$1"
    shift
    local joined
    local current_value="${!var_name-}"
    local IFS=:
    joined="$*"
    if [[ -n "${current_value}" ]]; then
        export "${var_name}=${joined}:${current_value}"
    else
        export "${var_name}=${joined}"
    fi
}

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

MODE="$1"
shift

REPEAT_COUNT=1
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--repeat" ]]; then
        if [[ $# -lt 2 ]]; then
            log_error "--repeat requires a positive integer value."
            usage
            exit 1
        fi
        REPEAT_COUNT="$2"
        shift 2
        continue
    fi
    FORWARD_ARGS+=("$1")
    shift
done

if [[ ! "${REPEAT_COUNT}" =~ ^[0-9]+$ ]] || [[ "${REPEAT_COUNT}" == "0" ]]; then
    log_error "Invalid --repeat value: ${REPEAT_COUNT}"
    usage
    exit 1
fi

case "${MODE}" in
    image)
        TARGET_NAME="modeling_private_qwen3_omni_modeling_qwen3_omni"
        ;;
    tts)
        TARGET_NAME="modeling_private_qwen3_omni_modeling_qwen3_omni_tts_min"
        ;;
    *)
        log_error "Unknown mode: ${MODE}"
        usage
        exit 1
        ;;
esac

if [[ ! -d "${OPENVINO_BIN}" ]]; then
    log_error "OpenVINO bin directory not found: ${OPENVINO_BIN}"
    exit 1
fi

TBB_DIR="$(find_tbb_dir || true)"
if [[ -z "${TBB_DIR}" ]]; then
    log_error "TBB runtime directory not found under ${ROOT_DIR}/openvino/temp"
    exit 1
fi

if [[ ! -d "${GENAI_DLL_DIR}" ]]; then
    log_error "OpenVINO GenAI runtime directory not found: ${GENAI_DLL_DIR}"
    exit 1
fi

if [[ ! -d "${GENAI_RUNTIME_BIN_DIR}" ]]; then
    log_error "OpenVINO GenAI runtime bin directory not found: ${GENAI_RUNTIME_BIN_DIR}"
    exit 1
fi

if [[ ! -f "${BRIDGE_DIR}/processing_qwen3_omni_bridge.py" ]]; then
    log_error "Qwen3 Omni bridge script not found: ${BRIDGE_DIR}/processing_qwen3_omni_bridge.py"
    exit 1
fi

TARGET_DIR="${GENAI_RELEASE_BIN_DIR}"
TARGET_EXE="${TARGET_DIR}/${TARGET_NAME}"
if [[ ! -f "${TARGET_EXE}" ]]; then
    TARGET_DIR="${GENAI_RUNTIME_BIN_DIR}"
    TARGET_EXE="${TARGET_DIR}/${TARGET_NAME}"
fi

if [[ ! -f "${TARGET_EXE}" ]]; then
    log_error "Executable not found: ${TARGET_NAME}"
    echo "        Reconfigure and rebuild openvino.genai after the modeling_private CMake hook is applied." >&2
    exit 1
fi

PATH_ENTRIES=()
if [[ -x "${VENV_BIN}/python" ]]; then
    PATH_ENTRIES+=("${VENV_BIN}")
else
    log_warn "${VENV_BIN}/python not found. Falling back to whatever 'python' resolves from PATH."
fi
PATH_ENTRIES+=(
    "${OPENVINO_BIN}"
    "${TBB_DIR}"
    "${GENAI_DLL_DIR}"
    "${GENAI_RUNTIME_BIN_DIR}"
    "${TARGET_DIR}"
)

prepend_env_path PATH "${PATH_ENTRIES[@]}"
prepend_env_path LD_LIBRARY_PATH "${PATH_ENTRIES[@]}"

export QWEN3_OMNI_BRIDGE_DIR="${BRIDGE_DIR}"
export OV_GENAI_USE_MODELING_API=1

if [[ "${REPEAT_COUNT}" != "1" ]]; then
    FORWARD_ARGS+=("--benchmark-runs" "${REPEAT_COUNT}")
fi

cd "${TARGET_DIR}"
printf '[RUN]'
printf ' %q' "${TARGET_EXE}" "${FORWARD_ARGS[@]}"
printf '\n'
"${TARGET_EXE}" "${FORWARD_ARGS[@]}"
