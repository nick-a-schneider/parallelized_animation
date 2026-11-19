from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp

##################################################
from .types import AnimationResult
from .writers import Mp4Writer

@dataclass
class AnimationRender:
    _temporary_directory: TemporaryDirectory
    chunk_paths: tuple[Path, ...]
    shape: tuple[int, int]
    frame_count: int
    worker_count: int
    elapsed_seconds: float
    _writer: Mp4Writer

    @property
    def temporary_directory(self) -> Path:
        return Path(self._temporary_directory.name)

    @property
    def frames_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0

        return self.frame_count / self.elapsed_seconds
    
    def save(self, output_path: str | Path) -> AnimationResult:
        """
        Render the animation and save the combined output.

        Worker-rendered MP4 chunks are concatenated without re-encoding. Only
        MP4 output is supported; other formats can be generated from the saved
        MP4 with separate conversion tools.

        Parameters
        ----------
        output_path : str | Path
            Destination file path.

        Returns
        -------
        AnimationResult
            Output path, frame count, worker count, and elapsed render time.

        Raises
        ------
        ValueError
            If the output path does not have an MP4 extension.
        """

        if self._temporary_directory is None:
            raise RuntimeError("Rendered MP4 chunks are unavailable.")

        output_path = Path(output_path).expanduser().resolve()
        if output_path.suffix.lower() != ".mp4":
            raise ValueError("Only MP4 output is supported")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        staging_path = self._create_staging_path(output_path)
        tmp_dir = Path(self._temporary_directory.name)

        try:
            self._writer.save_chunks(self.chunk_paths, staging_path, tmp_dir)
            os.replace(staging_path, output_path)

        finally:
            staging_path.unlink(missing_ok=True)

        return AnimationResult(
            output_path,
            self.frame_count,
            self.worker_count,
            self.elapsed_seconds
        )
        
    @staticmethod
    def _create_staging_path(output_path: Path) -> Path:
        """
        Reserve a unique staging path beside the final output file.

        The temporary file created by ``mkstemp`` is immediately removed so the
        writer receives a nonexistent path while retaining a collision-safe name.

        Parameters
        ----------
        output_path : Path
            Final output path.

        Returns
        -------
        Path
            Unique nonexistent path in the output directory.
        """
        
        descriptor, staging_name = mkstemp(
            prefix=f".{output_path.stem}_",
            suffix=output_path.suffix,
            dir=output_path.parent,
        )

        os.close(descriptor)

        staging_path = Path(staging_name)

        # Writers should create their output rather than receiving an existing
        # empty file.
        staging_path.unlink()

        return staging_path

    def __str__(self) -> str:
        return (
            f"Saved {self.frame_count:,} frames to {self.temporary_directory}\n"
            f"Workers: {self.worker_count}\n"
            f"Elapsed: {self.elapsed_seconds:.2f} s\n"
            f"Render rate: {self.frames_per_second:.2f} frames/s"
        )