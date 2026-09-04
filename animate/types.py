from __future__ import annotations

import os
from dataclasses import dataclass

from pathlib import Path
from typing import Callable, Generic, TypeVar

##################################################
from .scene import AnimationScene
from .writers.base_writer import AnimationWriter


FrameT = TypeVar("FrameT")

@dataclass(frozen=True)
class ParallelConfig:
    """
    Configure parallel animation rendering.

    Parameters
    ----------
    workers : int, optional
        Maximum number of worker processes. Defaults to at most four CPUs.
    temp_root : Path | None, optional
        Parent directory for temporary render files.
    """
    workers: int = min(4, os.cpu_count() or 1)
    temp_root: Path | None = None

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

        if self.temp_root is not None:
            object.__setattr__(self, "temp_root", Path(self.temp_root))


@dataclass(frozen=True)
class RenderChunk(Generic[FrameT]):
    """
    Describe one contiguous frame chunk assigned to a worker.

    Parameters
    ----------
    index : int
        Chunk index used to restore output order.
    frames : tuple[FrameT, ...]
        Frames assigned to the chunk.
    output_path : Path
        Intermediate file written for the chunk.
    """
    index: int
    frames: tuple[FrameT, ...]
    output_path: Path


@dataclass(frozen=True)
class SaveResult:
    """
    Summarize a completed animation save operation.

    Parameters
    ----------
    output_path : Path
        Final saved output path.
    frame_count : int
        Number of rendered frames.
    worker_count : int
        Number of workers used.
    elapsed_seconds : float
        Total elapsed render time in seconds.
    """
    output_path: Path
    frame_count: int
    worker_count: int
    elapsed_seconds: float

    @property
    def frames_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0

        return self.frame_count / self.elapsed_seconds
    
    def __str__(self) -> str:
        return (
            f"Saved {self.frame_count:,} frames to {self.output_path}\n"
            f"Workers: {self.worker_count}\n"
            f"Elapsed: {self.elapsed_seconds:.2f} s\n"
            f"Render rate: {self.frames_per_second:.2f} frames/s"
        )


@dataclass(frozen=True)
class AnimationJob(Generic[FrameT]):
    """
    Bundle callbacks and writer state required to render a frame chunk.

    Parameters
    ----------
    init_func : Callable[[], AnimationScene]
        Creates a worker-local animation scene.
    func : Callable[[FrameT, AnimationScene], None]
        Updates the scene for one frame.
    finalize_func : Callable[[AnimationScene], None] | None
        Optional callback run after a worker finishes rendering.
    writer : AnimationWriter
        Writer used to encode rendered frames.
    """

    init_func: Callable[[], AnimationScene]
    func: Callable[[FrameT, AnimationScene], None]
    finalize_func: Callable[[AnimationScene], None] | None
    writer: AnimationWriter
