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
    workers: int = min(4, os.cpu_count() or 1)
    temp_root: Path | None = None

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

        if self.temp_root is not None:
            object.__setattr__(self, "temp_root", Path(self.temp_root))


@dataclass(frozen=True)
class RenderChunk(Generic[FrameT]):
    index: int
    frames: tuple[FrameT, ...]
    output_path: Path


@dataclass(frozen=True)
class SaveResult:
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
    init_func: Callable[[], AnimationScene]
    func: Callable[[FrameT, AnimationScene], None]
    finalize_func: Callable[[AnimationScene], None] | None
    writer: AnimationWriter
