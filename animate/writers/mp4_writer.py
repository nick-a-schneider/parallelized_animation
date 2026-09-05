# mp4_writer.py

from __future__ import annotations

import shutil
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .base_writer import AnimationWriter


@dataclass(frozen=True)
class Mp4Writer(AnimationWriter):
    fps: int = 10
    ffmpeg: str = "ffmpeg"
    ffmpeg_threads: int = 1
    crf: int = 18
    preset: str = "medium"

    def __post_init__(self) -> None:
        if self.fps < 1:
            raise ValueError("fps must be at least 1")

        if self.ffmpeg_threads < 1:
            raise ValueError("ffmpeg_threads must be at least 1")

        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")

        if shutil.which(self.ffmpeg) is None:
            raise FileNotFoundError(
                f"Could not find FFmpeg executable: "
                f"{self.ffmpeg!r}"
            )

    def save_frames(
        self,
        frame_paths: Sequence[Path],
        output_path: Path,
        temporary_directory: Path,
    ) -> None:
        if not frame_paths:
            raise ValueError("Cannot save an animation with no rendered frames")

        input_pattern = temporary_directory / "frame_%08d.png"

        command = [
            self.ffmpeg,
            "-y",
            "-loglevel", "error",

            "-framerate", str(self.fps),
            "-start_number", "0",
            "-i", str(input_pattern),

            "-frames:v", str(len(frame_paths)),
            "-an",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-threads", str(self.ffmpeg_threads),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p",

            str(output_path),
        ]

        process = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed while writing "
                f"{output_path}:\n{process.stderr.strip()}"
            )