# mp4_writer.py

from __future__ import annotations

import shutil
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg

from ..scene import AnimationScene
from .base_writer import AnimationChunkWriter, AnimationWriter


@dataclass(frozen=True)
class Mp4Writer(AnimationWriter):
    fps: int = 10
    dpi: int = 150
    ffmpeg: str = "ffmpeg"
    ffmpeg_threads: int = 1
    crf: int = 18
    preset: str = "medium"
    supports_parallel = True

    def __post_init__(self) -> None:
        if self.fps < 1:
            raise ValueError("fps must be at least 1")

        if self.dpi < 1:
            raise ValueError("dpi must be at least 1")

        if self.ffmpeg_threads < 1:
            raise ValueError("ffmpeg_threads must be at least 1")

        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")

        if shutil.which(self.ffmpeg) is None:
            raise FileNotFoundError(
                f"Could not find FFmpeg executable: "
                f"{self.ffmpeg!r}"
            )

    @property
    def chunk_suffix(self) -> str:
        return ".mp4"

    def open_chunk(self, output_path: Path, scene: AnimationScene) -> AnimationChunkWriter:
        return Mp4ChunkWriter(output_path=output_path, scene=scene, config=self)

    def combine_chunks(self, chunk_paths: Sequence[Path], output_path: Path, temporary_directory: Path) -> None:
        if len(chunk_paths) == 1:
            shutil.move(str(chunk_paths[0]), str(output_path))
            return

        concat_path = temporary_directory / "chunks.ffconcat"

        with concat_path.open("w", encoding="utf-8") as stream:
            stream.write("ffconcat version 1.0\n")

            for chunk_path in chunk_paths:
                escaped = str(chunk_path.resolve()).replace("'", "'\\''")
                stream.write(f"file '{escaped}'\n")

        command = [
            self.ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_path),
            "-c", "copy",
            str(output_path)
        ]

        subprocess.run(command, check=True)


class Mp4ChunkWriter(AnimationChunkWriter):
    def __init__(self, *, output_path: Path, scene: AnimationScene, config: Mp4Writer) -> None:
        self.output_path = output_path
        self.config = config
        self.scene = scene
        self._finished = False
        
        self.figure = scene.figure
        self.figure.set_dpi(config.dpi)

        # MP4 output specifically requires an RGBA canvas.
        self.canvas = FigureCanvasAgg(self.figure)
        self.artists = tuple(scene.artists.values())

        # Dynamic artists are excluded from the cached background.
        for artist in self.artists:
            artist.set_animated(True)

        self.canvas.draw()

        self.background = self.canvas.copy_from_bbox(
            self.figure.bbox
        )

        self._process = self._start_encoder()

    def _start_encoder(self) -> subprocess.Popen[bytes]:
        width, height = self.canvas.get_width_height()
        command = [
            self.config.ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgba",
            "-s:v", f"{width}x{height}",
            "-r", str(self.config.fps),
            "-i",
            "-",
            "-an",
            "-c:v", "libx264",
            "-preset", self.config.preset,
            "-crf", str(self.config.crf),
            "-threads", str(self.config.ffmpeg_threads),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p",
            str(self.output_path)
        ]

        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, scene: AnimationScene) -> None:
        if self._finished:
            raise RuntimeError(
                "Cannot write to a finished MP4 chunk"
            )

        if scene is not self.scene:
            raise ValueError(
                "Chunk writer received a different scene"
            )

        if self._process.stdin is None:
            raise RuntimeError("FFmpeg stdin was not created")

        self.canvas.restore_region(self.background)

        for artist in self.artists:
            if artist.axes is not None:
                artist.axes.draw_artist(artist)
            else:
                self.figure.draw_artist(artist)

        self.canvas.blit(self.figure.bbox)

        self._process.stdin.write(
            memoryview(self.canvas.buffer_rgba())
        )

    def finish(self) -> None:
        if self._finished:
            return

        self._finished = True

        if self._process.stdin is None:
            raise RuntimeError("FFmpeg stdin was not created")

        self._process.stdin.close()

        stderr = (
            self._process.stderr.read().decode("utf-8", errors="replace")
            if self._process.stderr is not None
            else ""
        )

        return_code = self._process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed while writing "
                f"{self.output_path}:\n{stderr.strip()}"
            )

    def abort(self) -> None:
        if self._finished:
            return

        self._finished = True

        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
        except BaseException:
            pass

        try:
            if self._process.poll() is None:
                self._process.kill()

            self._process.wait()
        except BaseException:
            pass
