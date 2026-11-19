from __future__ import annotations

import shutil
import subprocess
import tempfile

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mp4Writer:
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
                f"Could not find FFmpeg executable: {self.ffmpeg!r}"
            )

    def chunk_command(self, output_path: Path, shape: tuple[int, int]) -> list[str]:
        width, height = shape

        return [
            self.ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "rgba",
            "-video_size", f"{width}x{height}",
            "-framerate", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-threads", str(self.ffmpeg_threads),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

    def save_chunks(self, chunk_paths: tuple[Path, ...], output_path: Path, temporary_directory: Path) -> None:
        if not chunk_paths:
            raise ValueError("Cannot save an animation with no rendered chunks")

        manifest_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="concat_",
                suffix=".txt",
                dir=temporary_directory,
                delete=False,
            ) as manifest:
                manifest_path = Path(manifest.name)

                for index, chunk_path in enumerate(chunk_paths):
                    if chunk_path.parent.resolve() != temporary_directory.resolve():
                        raise ValueError(
                            "MP4 chunks must be located in the render temporary directory"
                        )

                    expected_name = f"chunk_{index:08d}.mp4"
                    if chunk_path.name != expected_name:
                        raise ValueError(
                            f"Expected ordered MP4 chunk {expected_name!r}, "
                            f"got {chunk_path.name!r}"
                        )

                    if not chunk_path.is_file():
                        raise FileNotFoundError(
                            f"Rendered MP4 chunk does not exist: {chunk_path}"
                        )

                    manifest.write(f"file {chunk_path.name}\n")

            command = [
                self.ffmpeg,
                "-y",
                "-loglevel", "error",
                "-f", "concat",
                "-safe", "1",
                "-i", str(manifest_path),
                "-map", "0:v:0",
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed while assembling {output_path}:\n"
                    f"{result.stderr.strip()}"
                )

        finally:
            if manifest_path is not None:
                manifest_path.unlink(missing_ok=True)
