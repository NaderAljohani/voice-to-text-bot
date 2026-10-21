import logging
import os
import site
import sysconfig
from pathlib import Path


logger = logging.getLogger(__name__)
_DLL_DIRECTORY_HANDLES: list[object] = []
ADDED_CUDA_DLL_DIRS: list[Path] = []
ADDED_PATH_DIRS: list[Path] = []
MISSING_CUDA_DLL_DIRS: list[Path] = []


def _site_package_roots() -> list[Path]:
    roots: list[Path] = []
    for raw_path in getattr(site, "getsitepackages", lambda: [])():
        roots.append(Path(raw_path))

    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))

    for key in ("purelib", "platlib"):
        raw_path = sysconfig.get_paths().get(key)
        if raw_path:
            roots.append(Path(raw_path))

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved.name not in {"site-packages", "dist-packages"}:
            continue
        if resolved not in seen:
            unique_roots.append(resolved)
            seen.add(resolved)
    return unique_roots


def add_nvidia_dll_directories() -> None:
    relative_paths = (
        Path("nvidia") / "cuda_runtime" / "bin",
        Path("nvidia") / "cuda_nvrtc" / "bin",
        Path("nvidia") / "cublas" / "bin",
        Path("nvidia") / "cudnn" / "bin",
    )

    existing_dll_dirs: list[Path] = []
    site_roots = _site_package_roots()
    for relative_path in relative_paths:
        for site_root in site_roots:
            dll_dir = site_root / relative_path
            if dll_dir.is_dir():
                existing_dll_dirs.append(dll_dir)
            else:
                MISSING_CUDA_DLL_DIRS.append(dll_dir)
                logger.warning("CUDA DLL directory not found: %s", dll_dir)

    for dll_dir in existing_dll_dirs:
        if hasattr(os, "add_dll_directory"):
            try:
                handle = os.add_dll_directory(str(dll_dir))
            except OSError:
                logger.exception("Failed to add CUDA DLL directory: %s", dll_dir)
            else:
                _DLL_DIRECTORY_HANDLES.append(handle)
                ADDED_CUDA_DLL_DIRS.append(dll_dir)
                logger.warning("Added CUDA DLL directory: %s", dll_dir)
        else:
            logger.debug("os.add_dll_directory is not available on this platform.")

    prepend_existing_dirs_to_path(existing_dll_dirs)


def prepend_existing_dirs_to_path(dll_dirs: list[Path]) -> None:
    current_path = os.environ.get("PATH", "")
    current_parts = [part for part in current_path.split(os.pathsep) if part]
    current_parts_normalized = {_normalize_path_for_compare(Path(part)) for part in current_parts}

    additions: list[str] = []
    for dll_dir in dll_dirs:
        normalized = _normalize_path_for_compare(dll_dir)
        if normalized in current_parts_normalized:
            logger.debug("CUDA DLL directory already exists in PATH: %s", dll_dir)
            continue

        additions.append(str(dll_dir))
        ADDED_PATH_DIRS.append(dll_dir)
        current_parts_normalized.add(normalized)
        logger.warning("Prepended CUDA DLL directory to PATH: %s", dll_dir)

    if additions:
        os.environ["PATH"] = os.pathsep.join(additions + current_parts)


def _normalize_path_for_compare(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


if os.name == "nt":
    add_nvidia_dll_directories()
