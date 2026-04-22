from __future__ import annotations

import argparse
import filecmp
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class CopySource:
    name: str
    relative_path_template: str
    source_kind: str
    allowed_suffixes: tuple[str, ...] = ()
    include_executables: bool = False

    def resolve(self, workspace_root: Path, config: str) -> Path:
        return workspace_root / Path(self.relative_path_template.format(config=config))


WINDOWS_COPY_SOURCES = (
    CopySource(
        name="OpenVINO runtime DLL directory",
        relative_path_template="openvino/bin/intel64/{config}",
        source_kind="directory",
        allowed_suffixes=(".dll",),
    ),
    CopySource(
        name="OpenVINO GenAI DLL directory",
        relative_path_template="openvino.genai/build/openvino_genai",
        source_kind="directory",
        allowed_suffixes=(".dll",),
    ),
    CopySource(
        name="OpenVINO GenAI runtime bin directory",
        relative_path_template="openvino.genai/build/bin",
        source_kind="directory",
        allowed_suffixes=(".dll",),
    ),
    CopySource(
        name="OpenVINO GenAI executable directory",
        relative_path_template="openvino.genai/build/bin/{config}",
        source_kind="directory",
        allowed_suffixes=(".exe",),
    ),
    CopySource(
        name="TBB runtime DLL",
        relative_path_template="openvino/temp/Windows_AMD64/tbb/bin/tbb12.dll",
        source_kind="file",
        allowed_suffixes=(".dll",),
    ),
)


LINUX_COPY_SOURCES = (
    CopySource(
        name="OpenVINO runtime shared library directory",
        relative_path_template="openvino/bin/intel64/{config}",
        source_kind="directory",
        allowed_suffixes=(".so",),
    ),
    CopySource(
        name="OpenVINO GenAI shared library directory",
        relative_path_template="openvino.genai/build/openvino_genai",
        source_kind="directory",
        allowed_suffixes=(".so",),
    ),
    CopySource(
        name="OpenVINO GenAI runtime bin directory",
        relative_path_template="openvino.genai/build/bin",
        source_kind="directory",
        allowed_suffixes=(".so",),
        include_executables=True,
    ),
    CopySource(
        name="TBB runtime shared library directory",
        relative_path_template="openvino/temp/Linux_x86_64/tbb/lib",
        source_kind="directory",
        allowed_suffixes=(".so",),
    ),
)


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def format_bytes(size_in_bytes: int) -> str:
    size = float(size_in_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size_in_bytes} B"


def detect_platform_name() -> str:
    system_name = platform.system()
    if system_name == "Windows":
        return "windows"
    if system_name == "Linux":
        return "linux"
    raise RuntimeError(f"Unsupported platform: {system_name}")


def get_copy_sources(platform_name: str) -> tuple[CopySource, ...]:
    if platform_name == "windows":
        return WINDOWS_COPY_SOURCES
    if platform_name == "linux":
        return LINUX_COPY_SOURCES
    raise RuntimeError(f"Unsupported platform: {platform_name}")


def has_allowed_suffix(file: Path, allowed_suffixes: set[str]) -> bool:
    file_name_lower = file.name.lower()
    return any(
        suffix == ".so" and ".so" in file_name_lower or file_name_lower.endswith(suffix)
        for suffix in allowed_suffixes
    )


def is_platform_executable(file: Path, platform_name: str) -> bool:
    if not file.is_file():
        return False
    if platform_name == "windows":
        return file.suffix.lower() == ".exe"
    if platform_name == "linux":
        return file.suffix == "" and os.access(file, os.X_OK)
    return False


def is_matching_artifact(file: Path, source: CopySource, platform_name: str) -> bool:
    allowed_suffixes = {suffix.lower() for suffix in source.allowed_suffixes}
    return has_allowed_suffix(file, allowed_suffixes) or (
        source.include_executables and is_platform_executable(file, platform_name)
    )


def collect_source_files(
    source: CopySource,
    workspace_root: Path,
    config: str,
    platform_name: str,
) -> tuple[list[Path], list[str]]:
    resolved_path = source.resolve(workspace_root, config)

    if source.source_kind == "file":
        if not resolved_path.exists():
            return [], [
                f"{source.name}: expected file was not found: {resolved_path}"
            ]
        if not resolved_path.is_file():
            return [], [
                f"{source.name}: expected a file, but found a non-file path: {resolved_path}"
            ]
        if not is_matching_artifact(resolved_path, source, platform_name):
            return [], [
                f"{source.name}: file does not match expected artifact types for this platform: {resolved_path}"
            ]
        return [resolved_path], []

    if source.source_kind != "directory":
        return [], [f"{source.name}: unsupported source kind '{source.source_kind}'."]

    if not resolved_path.exists():
        return [], [
            f"{source.name}: expected directory was not found: {resolved_path}"
        ]
    if not resolved_path.is_dir():
        return [], [
            f"{source.name}: expected a directory, but found a non-directory path: {resolved_path}"
        ]

    matched_files = sorted(
        file
        for file in resolved_path.iterdir()
        if is_matching_artifact(file, source, platform_name)
    )
    if not matched_files:
        return [], [
            f"{source.name}: directory exists but no files matched expected artifact types for this platform: {resolved_path}"
        ]

    return matched_files, []


def copy_one_file(source_file: Path, destination_dir: Path) -> tuple[str, int]:
    destination_file = destination_dir / source_file.name
    file_size = source_file.stat().st_size

    if destination_file.exists():
        if not destination_file.is_file():
            raise RuntimeError(
                f"Destination exists but is not a file: {destination_file}"
            )

        if filecmp.cmp(source_file, destination_file, shallow=False):
            log(
                "SKIP",
                f"Identical file already exists, skipping: {source_file} -> {destination_file} ({format_bytes(file_size)})",
            )
            return "skipped", 0

        log(
            "COPY",
            f"Overwriting existing file: {source_file} -> {destination_file} ({format_bytes(file_size)})",
        )
        shutil.copy2(source_file, destination_file)
        return "overwritten", file_size

    log(
        "COPY",
        f"Copying file: {source_file} -> {destination_file} ({format_bytes(file_size)})",
    )
    shutil.copy2(source_file, destination_file)
    return "copied", file_size


def collect_package_files(destination_dir: Path, platform_name: str) -> list[Path]:
    if not destination_dir.exists():
        return []

    package_filter = CopySource(
        name="packaged artifact",
        relative_path_template="",
        source_kind="directory",
        allowed_suffixes=(".dll", ".exe") if platform_name == "windows" else (".so",),
        include_executables=platform_name == "linux",
    )

    return sorted(
        file
        for file in destination_dir.iterdir()
        if is_matching_artifact(file, package_filter, platform_name)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect built platform-specific runtime artifacts from the OpenVINO and OpenVINO GenAI "
            "workspace into a single package directory."
        ),
        epilog=(
            "Default behavior:\n"
            "  - Use Release unless --build-type is specified.\n"
            "  - Use the workspace-level 'package' directory as the output root.\n"
            "  - Store files in '<output_root>/<Config>'.\n"
            "  - Keep existing files unless --clean is specified.\n\n"
            "Examples:\n"
            "  python scripts\\package.py\n"
            "  python scripts\\package.py --build-type RelWithDebInfo\n"
            "  python scripts\\package.py --clean\n"
            "  python scripts\\package.py --output D:\\artifacts\\package_bundle\n"
            "  python scripts\\package.py --output custom_package --clean\n"
            "  python3 scripts/package.py --output /tmp/package_bundle --clean"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete existing packaged artifacts in the final destination directory before copying. "
            "Only files directly inside '<output_root>/<Config>' are removed."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Override the package output root directory. If omitted, the default is "
            "the workspace-level 'package' directory. Relative paths are resolved "
            "relative to the workspace root (two levels above this script)."
        ),
    )
    parser.add_argument(
        "--build-type",
        choices=("Release", "RelWithDebInfo"),
        default="Release",
        help=(
            "Build configuration name used for config-sensitive source paths. "
            "Defaults to Release."
        ),
    )
    return parser


def resolve_output_root(output_arg: str | None, workspace_root: Path) -> Path:
    if not output_arg:
        return workspace_root / "package"

    output_path = Path(output_arg)
    if not output_path.is_absolute():
        output_path = workspace_root / output_path
    return output_path.resolve()


def clean_destination_dir(destination_dir: Path, platform_name: str) -> tuple[int, int]:
    if not destination_dir.exists():
        log("INFO", f"Clean requested but destination does not exist yet: {destination_dir}")
        return 0, 0

    removed_files = 0
    removed_bytes = 0

    for file in collect_package_files(destination_dir, platform_name):
        file_size = file.stat().st_size
        log("CLEAN", f"Removing existing package file: {file} ({format_bytes(file_size)})")
        file.unlink()
        removed_files += 1
        removed_bytes += file_size

    return removed_files, removed_bytes


def patch_linux_runpaths(packaged_files: list[Path]) -> tuple[int, list[str]]:
    patchelf_path = shutil.which("patchelf")
    if not patchelf_path:
        return 0, [
            "Linux packaging requires 'patchelf' in PATH to make copied ELF artifacts relocatable."
        ]

    patched_files = 0
    issues: list[str] = []
    for artifact in packaged_files:
        try:
            subprocess.run(
                [patchelf_path, "--set-rpath", "$ORIGIN", str(artifact)],
                check=True,
                capture_output=True,
                text=True,
            )
            patched_files += 1
        except subprocess.CalledProcessError as error:
            stderr_output = error.stderr.strip()
            details = f": {stderr_output}" if stderr_output else ""
            issues.append(f"Failed to patch RUNPATH for '{artifact}'{details}")

    return patched_files, issues


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    workspace_root = repo_root.parent
    package_root = resolve_output_root(args.output, workspace_root)
    chosen_config = args.build_type
    platform_name = detect_platform_name()
    copy_sources = get_copy_sources(platform_name)

    log("INFO", f"Script directory: {script_dir}")
    log("INFO", f"Repository root: {repo_root}")
    log("INFO", f"Workspace root: {workspace_root}")
    log("INFO", f"Configured package root: {package_root}")
    log("INFO", f"Selected configuration: {chosen_config}")
    log("INFO", f"Detected platform: {platform_name}")

    destination_dir = package_root / chosen_config
    destination_dir.mkdir(parents=True, exist_ok=True)
    log("INFO", f"Package output directory: {destination_dir}")

    summary = {
        "matched_files": 0,
        "copied_files": 0,
        "overwritten_files": 0,
        "skipped_files": 0,
        "patched_files": 0,
        "error_count": 0,
        "matched_bytes": 0,
        "written_bytes": 0,
        "cleaned_files": 0,
        "cleaned_bytes": 0,
    }

    if args.clean:
        cleaned_files, cleaned_bytes = clean_destination_dir(destination_dir, platform_name)
        summary["cleaned_files"] = cleaned_files
        summary["cleaned_bytes"] = cleaned_bytes
        log(
            "INFO",
            f"Clean completed. Removed {cleaned_files} file(s), reclaimed {format_bytes(cleaned_bytes)}.",
        )

    for source in copy_sources:
        resolved_path = source.resolve(workspace_root, chosen_config)
        log("INFO", f"Scanning source '{source.name}': {resolved_path}")
        source_files, source_issues = collect_source_files(source, workspace_root, chosen_config, platform_name)

        if source_issues:
            summary["error_count"] += len(source_issues)
            for issue in source_issues:
                log("ERROR", issue)
            continue

        log(
            "INFO",
            f"Found {len(source_files)} file(s) from '{source.name}'.",
        )

        for source_file in source_files:
            summary["matched_files"] += 1
            file_size = source_file.stat().st_size
            summary["matched_bytes"] += file_size

            try:
                action, written_bytes = copy_one_file(source_file, destination_dir)
            except Exception as error:  # pragma: no cover - defensive logging
                summary["error_count"] += 1
                log(
                    "ERROR",
                    f"Failed to copy file '{source_file}' to '{destination_dir}': {error}",
                )
                continue

            if action == "copied":
                summary["copied_files"] += 1
                summary["written_bytes"] += written_bytes
            elif action == "overwritten":
                summary["overwritten_files"] += 1
                summary["written_bytes"] += written_bytes
            else:
                summary["skipped_files"] += 1

    packaged_files = collect_package_files(destination_dir, platform_name)
    if platform_name == "linux" and packaged_files:
        patched_files, patch_issues = patch_linux_runpaths(packaged_files)
        summary["patched_files"] = patched_files
        if patch_issues:
            summary["error_count"] += len(patch_issues)
            for issue in patch_issues:
                log("ERROR", issue)
        else:
            log("INFO", f"Patched Linux RUNPATH to $ORIGIN for {patched_files} artifact(s).")

    packaged_bytes = sum(file.stat().st_size for file in packaged_files)

    log("SUMMARY", "Packaging finished.")
    log("SUMMARY", f"Configuration: {chosen_config}")
    log("SUMMARY", f"Destination: {destination_dir}")
    log(
        "SUMMARY",
        f"Clean option: {'enabled' if args.clean else 'disabled'}; cleaned files: {summary['cleaned_files']}; cleaned size: {format_bytes(summary['cleaned_bytes'])}",
    )
    log("SUMMARY", f"Matched source files: {summary['matched_files']}")
    log(
        "SUMMARY",
        f"Copied files: {summary['copied_files']}, overwritten files: {summary['overwritten_files']}, skipped identical files: {summary['skipped_files']}",
    )
    if platform_name == "linux":
        log("SUMMARY", f"Patched Linux RUNPATH entries: {summary['patched_files']}")
    log("SUMMARY", f"Encountered errors: {summary['error_count']}")
    log(
        "SUMMARY",
        f"Total matched source size: {format_bytes(summary['matched_bytes'])}",
    )
    log(
        "SUMMARY",
        f"Total bytes written this run: {format_bytes(summary['written_bytes'])}",
    )
    log(
        "SUMMARY",
        f"Current package folder contains {len(packaged_files)} packaged artifact(s), total size {format_bytes(packaged_bytes)}.",
    )

    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
