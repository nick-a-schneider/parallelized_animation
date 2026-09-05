from __future__ import annotations

import os
from tempfile import TemporaryDirectory, mkstemp
import time

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Generic, Iterable

##################################################
from .scene import AnimationScene
from .progress import _LocalProgress
from .worker_helpers import render_animation_chunk, wait_for_workers
from .types import FrameT, RenderConfig, RenderChunk, AnimationResult, AnimationJob


from .writers import AnimationWriter, make_writer, WRITERS

class ParallelAnimation(Generic[FrameT]):
    """
    Render a finite frame sequence using worker-local animation scenes.

    Frames are divided into contiguous chunks. Each worker creates one scene,
    reuses it for its chunk, and writes an intermediate file that is later
    combined by the selected writer.

    Parameters
    ----------
    frames : Iterable[FrameT]
        Frame values passed to ``func``.
    init_func : Callable[[], AnimationScene]
        Creates a worker-local animation scene.
    func : Callable[[FrameT, AnimationScene], None]
        Updates the scene for one frame.
    finalize_func : Callable[[AnimationScene], None] | None, optional
        Finalizes a worker scene after its chunk is rendered.
    config : ParallelConfig | None, optional
        Parallel rendering configuration. Uses defaults when omitted.
    """

    def __init__(self,
        frames: Iterable[FrameT],
        init_func: Callable[[], AnimationScene],
        func: Callable[[FrameT, AnimationScene], None],
        *,
        finalize_func: Callable[[AnimationScene], None] | None = None
    ) -> None:
        """
        Initialize an animation render job.

        Raises
        ------
        TypeError
            If ``init_func``, ``func``, or a provided ``finalize_func`` is not
            callable.
        """
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
        self._frames: tuple[FrameT, ...] | None = None
        
        
        self._temporary_directory: TemporaryDirectory | None = None
        self._rendered_frames: tuple[Path, ...] = ()
        self._render_worker_count = 0
        self._render_elapsed = 0.0

    @property
    def frames(self) -> tuple[FrameT, ...]:
        """
        Materialize and cache the input frame iterable.

        Returns
        -------
        tuple[FrameT, ...]
            Cached frame sequence.

        Raises
        ------
        ValueError
            If the frame iterable is empty.
        """
        if self._frames is None:
            self._frames = tuple(self._frame_source)

            if not self._frames:
                raise ValueError(
                    "ParallelAnimation requires at least one frame"
                )

        return self._frames
    def _clear_render(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()

        self._temporary_directory = None
        self._rendered_frames = ()
        self._render_worker_count = 0
        self._render_elapsed = 0.0
    
    def render(self, dpi: int, worker_count: int = 0, temp_root: Path | None = None) -> AnimationResult:
        self._clear_render()
        frames = self.frames
        config = RenderConfig(dpi, worker_count, temp_root)

        chunk_count = config.workers

        temporary_directory = TemporaryDirectory(None, "par_anim_", config.temp_root)

        self._temporary_directory = temporary_directory
        tmp_dir = Path(temporary_directory.name)

        job = AnimationJob(self._init_func, self._func, self._finalize_func, config)

        chunks = self._partition_frames(frames, chunk_count, tmp_dir)

        started_at = time.perf_counter()

        try:
            self._render_chunks(job, chunks, worker_count)

        except BaseException:
            self._clear_render()
            raise

        self._render_elapsed = time.perf_counter() - started_at
        self._render_worker_count = worker_count

        self._rendered_frames = tuple(
            tmp_dir / f"frame_{index:08d}.png"
            for index in range(len(frames))
        )
        
        return AnimationResult(
            Path(temporary_directory.name),
            len(self.frames),
            self._render_worker_count,
            self._render_elapsed,
        )
    
    def save(self, output_path: str | Path, *, writer: Any | None = None, **writer_options: Any) -> AnimationResult:
        """
        Render the animation and save the combined output.

        The writer is inferred from the output extension unless explicitly
        provided. Rendering occurs in temporary chunk files and the requested
        output path is replaced only after successful completion.

        Parameters
        ----------
        output_path : str | Path
            Destination file path.
        writer : AnimationWriter | None, optional
            Explicit writer. If omitted, one is selected from ``WRITERS``.
        **writer_options : Any
            Options passed to the inferred writer factory.

        Returns
        -------
        SaveResult
            Output path, frame count, worker count, and elapsed render time.

        Raises
        ------
        ValueError
            If no writer is registered for the output extension.
        """
        if not self._rendered_frames:
            raise RuntimeError(
                "Animation has not been rendered. Call render() before save()."
            )

        if self._temporary_directory is None:
            raise RuntimeError("Rendered frame data is unavailable.")

        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selected_writer = self._resolve_writer(output_path, writer, writer_options)

        staging_path = self._create_staging_path(output_path)
        tmp_dir = Path(self._temporary_directory.name)

        try:
            selected_writer.save_frames(self._rendered_frames, staging_path, tmp_dir)
            os.replace(staging_path, output_path)

        finally:
            staging_path.unlink(missing_ok=True)

        return AnimationResult(
            output_path,
            len(self.frames),
            self._render_worker_count,
            self._render_elapsed,
        )

    def _resolve_writer(self, 
        output_path: Path, 
        writer: AnimationWriter | None,
        writer_options: dict[str, Any]
    )-> AnimationWriter:
        """
        Select an explicit writer or construct one from the file extension.

        Parameters
        ----------
        output_path : Path
            Destination path used for extension inference.
        writer : AnimationWriter | None
            Explicit writer, or ``None`` to infer one.
        writer_options : dict[str, Any]
            Options passed to the inferred writer factory.

        Returns
        -------
        AnimationWriter
            Writer used for rendering and chunk combination.

        Raises
        ------
        ValueError
            If no writer is registered for the output extension.
        """
        
        if writer is not None:
            return writer

        extension = output_path.suffix.lower()
        
        if extension in WRITERS:
            return make_writer(WRITERS[extension], **writer_options)
        
        raise ValueError(
            f"No default writer for output extension {extension!r}; "
            "provide writer= explicitly"
        ) 

    def _partition_frames(
        self,
        frames: tuple[FrameT, ...],
        chunk_count: int,
        temporary_directory: Path,
    ) -> list[RenderChunk[FrameT]]:
        quotient, remainder = divmod(len(frames), chunk_count)

        chunks: list[RenderChunk[FrameT]] = []
        begin = 0

        for index in range(chunk_count):
            size = quotient + (index < remainder)
            end = begin + size

            chunks.append(
                RenderChunk(
                    index,
                    begin,
                    frames[begin:end],
                    temporary_directory,
                )
            )

            begin = end

        return chunks

    def _render_chunks(
        self,
        job: AnimationJob[FrameT],
        chunks: list[RenderChunk[FrameT]],
        worker_count: int,
    ) -> list[int]:
        total_frames = sum(len(chunk.frames) for chunk in chunks)

        if worker_count == 1:
            progress = _LocalProgress(total_frames)

            return [
                render_animation_chunk(job, chunk, progress)
                for chunk in chunks
            ]

        context = get_context("spawn")

        with context.Manager() as manager:
            progress_queue = manager.Queue()

            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
            ) as executor:
                futures = [
                    executor.submit(
                        render_animation_chunk,
                        job,
                        chunk,
                        progress_queue,
                    )
                    for chunk in chunks
                ]

                try:
                    return wait_for_workers(
                        futures,
                        progress_queue,
                        total_frames,
                    )

                except BaseException:
                    for future in futures:
                        future.cancel()

                    raise
            

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        """
        Normalize a file extension to lowercase ``.suffix`` form.

        Parameters
        ----------
        extension : str
            Extension with or without a leading period.

        Returns
        -------
        str
            Normalized extension.

        Raises
        ------
        ValueError
            If ``extension`` is empty or whitespace.
        """
        extension = extension.strip().lower()

        if not extension:
            raise ValueError("Output filename must have an extension")

        if not extension.startswith("."):
            extension = f".{extension}"

        return extension

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

