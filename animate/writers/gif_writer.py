from __future__ import annotations

import shutil
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .base_writer import AnimationWriter


@dataclass(frozen=True)
class GifWriter(AnimationWriter):
    fps: int = 10
    ffmpeg: str = "ffmpeg"

    def __post_init__(self) -> None:
        if self.fps < 1:
            raise ValueError("fps must be at least 1")

        if shutil.which(self.ffmpeg) is None:
            raise FileNotFoundError(
                f"Could not find FFmpeg executable: "
                f"{self.ffmpeg!r}"
            )

    def save_frames(self, frame_paths: Sequence[Path], output_path: Path, temporary_directory: Path) -> None:
        if not frame_paths:
            raise ValueError(
                "Cannot save an animation with no rendered frames"
            )

        input_pattern = temporary_directory / "frame_%08d.png"
        palette_path = temporary_directory / "palette.png"

        palette_command = [
            self.ffmpeg,
            "-y",
            "-loglevel", "error",
            "-framerate", str(self.fps),
            "-start_number", "0",
            "-i", str(input_pattern),
            "-frames:v", str(len(frame_paths)),
            "-vf", "palettegen",
            str(palette_path),
        ]

        self._run(palette_command, "generating GIF palette")

        gif_command = [
            self.ffmpeg,
            "-y",
            "-loglevel", "error",
            "-framerate", str(self.fps),
            "-start_number", "0",
            "-i", str(input_pattern),
            "-i", str(palette_path),
            "-frames:v", str(len(frame_paths)),
            "-lavfi", "paletteuse",
            str(output_path),
        ]

        self._run(gif_command, "writing GIF")

    @staticmethod
    def _run(command: list[str], operation: str) -> None:
        process = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed while {operation}:\n"
                f"{process.stderr.strip()}"
            )