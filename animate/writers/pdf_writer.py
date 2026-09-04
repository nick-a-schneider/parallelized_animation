# pdf_writer.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from ..scene import AnimationScene
from .base_writer import AnimationChunkWriter, AnimationWriter

@dataclass(frozen=True)
class PdfWriter(AnimationWriter):
    dpi: int = 150
    supports_parallel = False

    def __post_init__(self) -> None:
        if self.dpi < 1:
            raise ValueError("dpi must be at least 1")

    @property
    def chunk_suffix(self) -> str:
        return ".pdf"

    def open_chunk(self, output_path: Path, scene: AnimationScene) -> AnimationChunkWriter:
        return PdfChunkWriter(output_path=output_path, scene=scene, dpi=self.dpi)

class PdfChunkWriter(AnimationChunkWriter):
    def __init__(self, output_path: Path, scene: AnimationScene, dpi: int) -> None:
        self.scene = scene
        self.dpi = dpi
        self.pdf = PdfPages(output_path)
        self._finished = False

    def write_frame(self, scene: AnimationScene) -> None:
        if self._finished:
            raise RuntimeError("Cannot write to a finished PDF")

        if scene is not self.scene:
            raise ValueError("PDF writer received a different scene")

        self.pdf.savefig(scene.figure, dpi=self.dpi)

    def finish(self) -> None:
        if self._finished:
            return

        try:
            self.pdf.close()
        finally:
            self._finished = True

    def abort(self) -> None:
        if self._finished:
            return
        try:
            self.pdf.close()
        except BaseException:
            pass
        finally:
            self._finished = True
