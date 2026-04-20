#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENVINO_DIR="${ROOT_DIR}/openvino"
GENAI_DIR="${ROOT_DIR}/openvino.genai"
OPENVINO_URL="https://github.com/Yanfeng-Mi/openvino.git"
GENAI_URL="https://github.com/Yanfeng-Mi/openvino.genai.git"
OPENVINO_REF="2026.1.0"
GENAI_REF="2026.1.0.0"

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

ensure_git() {
    if ! command -v git >/dev/null 2>&1; then
        log_error "git not found in PATH."
        return 1
    fi
}

ensure_repo() {
    local repo_dir="$1"
    local repo_url="$2"
    local repo_ref="$3"
    local repo_name="$4"

    if [[ -d "${repo_dir}/.git" ]]; then
        log_info "${repo_name} already exists: ${repo_dir}"
    else
        if [[ -e "${repo_dir}" ]]; then
            log_error "${repo_name} path exists but is not a git repo: ${repo_dir}"
            return 1
        fi

        echo "[CLONE] ${repo_name}"
        git clone "${repo_url}" "${repo_dir}"
    fi

    echo "[CHECKOUT] ${repo_name} -> ${repo_ref}"
    git -C "${repo_dir}" fetch --all --tags
    git -C "${repo_dir}" checkout "${repo_ref}"

    echo "[SUBMODULE] ${repo_name}"
    git -C "${repo_dir}" submodule update --init --recursive
}

main() {
    log_info "repo.sh path      : ${SCRIPT_DIR}"
    log_info "workspace root    : ${ROOT_DIR}"

    ensure_git
    ensure_repo "${OPENVINO_DIR}" "${OPENVINO_URL}" "${OPENVINO_REF}" "openvino"
    ensure_repo "${GENAI_DIR}" "${GENAI_URL}" "${GENAI_REF}" "openvino.genai"

    echo "[OK] Repository setup finished."
    echo "     Ready to build from: ${ROOT_DIR}"
}

main "$@"