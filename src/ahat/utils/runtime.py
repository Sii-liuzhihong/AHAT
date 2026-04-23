"""
Runtime Utilities
=================

This module provides low-level runtime support functions including:
1. File system helpers: path preparation, MIME type detection, and unified directory traversalw.
2. Execution helpers: Execution timing decorators and signal handling for clean termination of
multi-threaded processes.

Architecture Position:
    [Shared Infrastructure / Utility] -> Basic runtime utilities.
"""

import mimetypes
import os
import signal
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, ParamSpec, TypeVar

from ahat.schemas.typings import Paths

__all__ = [
    "setup_path",
    "calculate_running_time",
    "register_force_exit",
    "get_sorted_paths",
    "get_mime_type",
    "is_image_file",
    "is_video_file"
]

P = ParamSpec("P")
R = TypeVar("R")


def setup_path(path: Paths, is_file: bool = False, mkdir: bool = True) -> Path:
    """Resolve a path and optionally create its parent directories.

    Args:
        path (Paths): The input path (string or Path object).
        is_file (bool): If True, treats the path as a file and creates directories
            for its parent. If False, treats the path as a directory. Defaults to False.
        mkdir (bool): If True, creates the directory structure. Defaults to True.
            Generally, False for input paths and True for output paths.

    Returns:
        Path: The resolved absolute Path object.

    Raises:
        ValueError: If the input path is None.
    """

    if path is None:
        raise ValueError("Path cannot be None.")
    
    p = Path(path).resolve()

    if mkdir:
        target_dir = p.parent if is_file else p
        target_dir.mkdir(parents=True, exist_ok=True)

    return p


def get_sorted_paths(
    directory: Paths,
    filter_func: Optional[Callable[[Path], bool]] = None,
    sort_key: Optional[Callable[[Path], Any]] = None,
    ignore_hidden: bool = True
) -> List[Path]:
    """List and sort contents of a directory with custom filtering and sorting.

    Default Sorting Logic:
        - If ALL filtered items have numeric names (e.g., "0.jpg", "10.jpg" or dirs "0", "1"),
          they are sorted numerically.
        - Otherwise, they are sorted alphabetically (natural dictionary order).

    Args:
        directory (Paths): The target directory.
        filter_func (Callable, optional): A function taking a Path and returning bool.
            Only items where filter_func(p) is True are included.
            Example: `lambda p: p.is_dir()`, `lambda p: p.suffix == ".json"`.
        sort_key (Callable, optional): A custom key function for sorting.
            If provided, it overrides the default sorting logic.
        ignore_hidden (bool): If True, skips files starting with ".", like ".DS_Store" in MacOS. Defaults to True.

    Returns:
        List[Path]: A sorted list of Path objects.
    """
    directory = setup_path(directory, mkdir=False)
    candidates: List[Path] = []

    # Collection & Filtering
    for p in directory.iterdir():
        # Check hidden files
        if ignore_hidden and p.name.startswith("."):
            continue
        
        # Apply custom filter
        if filter_func and not filter_func(p):
            continue
            
        candidates.append(p)

    if not candidates:
        return []
    
    # Sorting
    # Priority 1: User-provided sort key
    if sort_key:
        candidates.sort(key=sort_key)
        return candidates


    # Priority 2: Default sorting strategy: if ALL candidates look like numbers, use numeric sort.
    # Check if stem (filename without ext) is numeric for files, or name is numeric for dirs
    all_numeric = all(p.stem.isdigit() if p.is_file() else p.name.isdigit() for p in candidates)

    if all_numeric:
        # Numeric Sort: "0.jpg", "2.jpg", "10.jpg"
        candidates.sort(key=lambda x: int(x.stem) if x.is_file() else int(x.name))
    else:
        # Dictionary Sort: "0.jpg", "10.jpg", "2.jpg" (or standard alphabetical)
        candidates.sort(key=lambda x: x.name)

    return candidates


def get_mime_type(path: Paths) -> str:
    """Get the mime type ("image/jpeg", "video/mp4", etc.) from a file path."""
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or ""


def is_image_file(path: Paths) -> bool:
    """Check if the path points to an image file."""
    p = setup_path(path, is_file=True, mkdir=False)
    return p.is_file() and get_mime_type(p).startswith("image/")


def is_video_file(path: Paths) -> bool:
    """Check if the path points to a video file."""
    p = setup_path(path, is_file=True, mkdir=False)
    return p.is_file() and get_mime_type(p).startswith("video/")


def calculate_running_time(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to measure and print the execution time of a function.
    
    Args:
        func (Callable): The function to wrap.
        
    Returns:
        Callable: The wrapped function.
        
    Example:
        (1)
        @calculate_running_time
        def func():
            ...

        func()

        (2)
        func = calculate_running_time(func)
        func()

    """
    
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()

        taken_time = (end - start) / 1e9
        print(f"[TIMER] {func.__name__} finished in {taken_time:.4f}s")
        return result
    
    return wrapper


def register_force_exit() -> None:
    """Register signal handlers to force immediate process termination.

    This is particularly useful for multi-threaded applications where
    standard "sys.exit()" might hang waiting for daemon threads to join.
    It handles SIGINT (Ctrl+C) and SIGTERM.
    """

    def force_exit_handler(signum: int, frame) -> None:
        # Get signal name safely
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = str(signum)
        
        print(f"\n🛑 Signal {sig_name} received. Forcing immediate exit via os._exit(1).")

        # os._exit kills the process immediately without cleanup
        # (necessary to kill stuck threads)
        os._exit(1)

    # Register signal handlers for shutdown
    signal.signal(signal.SIGINT, force_exit_handler)

    # SIGTERM is not available on Windows
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, force_exit_handler)
