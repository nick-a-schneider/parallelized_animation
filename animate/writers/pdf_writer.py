from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from .base_writer import AnimationWriter


@dataclass(frozen=True)
class PdfWriter(AnimationWriter):

    def save_frames(self, frame_paths: Sequence[Path], output_path: Path, temporary_directory: Path) -> None:
        if not frame_paths:
            raise ValueError("Cannot save an animation with no rendered frames")

        images: list[Image.Image] = []

        try:
            for frame_path in frame_paths:
                image = Image.open(frame_path).convert("RGB")
                images.append(image)

            first, *rest = images

            first.save(output_path, format="PDF", save_all=True, append_images=rest)

        finally:
            for image in images:
                image.close()