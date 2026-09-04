from __future__ import annotations

import os
from tempfile import TemporaryDirectory, mkstemp
import time

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, Iterable

##################################################
from .scene import AnimationScene
from .progress import _LocalProgress
from .worker_helpers import render_animation_chunk, wait_for_workers
from .types import FrameT, ParallelConfig, RenderChunk, SaveResult, AnimationJob


from .writers import AnimationWriter, Mp4Writer, PdfWriter, make_writer, WRITERS

class ParallelAnimation(Generic[FrameT]):
    """
    Render a finite sequence of frames using worker-local animation scenes.

    Each worker creates one scene and reuses it for every frame in its
    contiguous frame chunk.
    """

    _writer_registry: ClassVar[ dict[str, Callable[..., Any]] ] = {}

    def __init__(self,
        frames: Iterable[FrameT],
        init_func: Callable[[], AnimationScene],
        func: Callable[[FrameT, AnimationScene], None],
        *,
        finalize_func: Callable[[AnimationScene], None] | None = None,
        config: ParallelConfig | None = None
    ) -> None:
        if not callable(init_func):
            raise TypeError("init_func must be callable")

        if not callable(func):
            raise TypeError("func must be callable")

        if finalize_func is not None and not callable(finalize_func):
            raise TypeError("finalize_func must be callable or None")

        self._frame_source = frames
        self._init_func = init_func
        self._func = func
        self._finalize_func = finalize_func
        self._config = config or ParallelConfig()

        self._frames: tuple[FrameT, ...] | None = None

    @property
    def config(self) -> ParallelConfig:
        return self._config

    @property
    def frames(self) -> tuple[FrameT, ...]:
        """
        Materialize and cache the frame iterable.

        This allows one animation to be saved multiple times even when its
        original frame source was a generator.
        """
        if self._frames is None:
            self._frames = tuple(self._frame_source)

            if not self._frames:
                raise ValueError(
                    "ParallelAnimation requires at least one frame"
                )

        return self._frames

    def save(self, output_path: str | Path, *, writer: Any | None = None, **writer_options: Any) -> SaveResult:
        """
        Render the animation and save it through an explicit or inferred writer.

        If writer is omitted, its type is inferred from output_path's suffix.
        Writer options are passed to the registered writer factory.
        """
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selected_writer = self._resolve_writer(output_path, writer, writer_options)

        frames = self.frames
        worker_count = (
            min(self.config.workers, len(frames))
            if selected_writer.supports_parallel
            else 1
        )

        job = AnimationJob(self._init_func, self._func, self._finalize_func, selected_writer)

        started_at = time.perf_counter()
        staging_path = self._create_staging_path(output_path)

        try:
            with TemporaryDirectory(None, "par_anim_",self.config.temp_root) as temporary_directory:
                tmp_dir = Path(temporary_directory)
                chunk_suffix = selected_writer.chunk_suffix
                
                chunks = self._partition_frames(frames, worker_count, tmp_dir, chunk_suffix)

                rendered_chunks = self._render_chunks(job, chunks, worker_count)

                chunk_paths = [ chunk_path for _, chunk_path in rendered_chunks]
                
                selected_writer.combine_chunks(chunk_paths, staging_path, tmp_dir)

            # The requested output is touched only after rendering and
            # concatenation have completed successfully.
            os.replace(staging_path, output_path)

        finally:
            staging_path.unlink(missing_ok=True)

        elapsed = time.perf_counter() - started_at
        
        return SaveResult(output_path, len(frames), worker_count, elapsed)

    def _resolve_writer(self, 
        output_path: Path, 
        writer: AnimationWriter | None,
        writer_options: dict[str, Any]
    )-> AnimationWriter:
        
        if writer is not None:
            return writer

        extension = output_path.suffix.lower()
        
        if extension in WRITERS:
            return make_writer(WRITERS[extension], **writer_options)
        
        raise ValueError(
            f"No default writer for output extension {extension!r}; "
            "provide writer= explicitly"
        ) 

    def _partition_frames(self,
        frames: tuple[FrameT, ...],
        worker_count: int,
        temporary_directory: Path,
        chunk_suffix: str,
    ) -> list[RenderChunk[FrameT]]:
        
        quotient, remainder = divmod(len(frames), worker_count)

        chunks: list[RenderChunk[FrameT]] = []
        begin = 0

        for index in range(worker_count):
            size = quotient + (index < remainder)
            end = begin + size
            output_path = temporary_directory / f"chunk_{index:04d}{chunk_suffix}"
            chunks.append(RenderChunk(index, frames[begin:end], output_path))

            begin = end

        return chunks

    def _render_chunks(self,
        job: AnimationJob[FrameT],
        chunks: list[RenderChunk[FrameT]],
        worker_count: int,
    ) -> list[tuple[int, Path]]:
        
        total_frames = sum(len(chunk.frames) for chunk in chunks)
        
        if worker_count == 1:
            progress = _LocalProgress(total_frames)
            return [render_animation_chunk(job, chunks[0], progress)]

        context = get_context("spawn")
        
        with context.Manager() as manager:
            progress_queue = manager.Queue()

            with ProcessPoolExecutor(worker_count, context) as executor:
                futures = [
                    executor.submit(render_animation_chunk, job, chunk, progress_queue)
                    for chunk in chunks
                ]

                try:
                    return wait_for_workers(futures, progress_queue, total_frames)

                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        extension = extension.strip().lower()

        if not extension:
            raise ValueError("Output filename must have an extension")

        if not extension.startswith("."):
            extension = f".{extension}"

        return extension

    @staticmethod
    def _create_staging_path(output_path: Path) -> Path:
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

