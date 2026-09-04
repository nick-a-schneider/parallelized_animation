from __future__ import annotations

import time
import queue

from concurrent.futures import Future
from pathlib import Path
from typing import Callable, Sequence


import matplotlib.pyplot as plt

##################################################
from .scene import AnimationScene
from .progress import _LocalProgress, print_status
from .types import FrameT, RenderChunk, AnimationJob


def finalize_animation_scene(
    scene: AnimationScene,
    finalize_func: Callable[[AnimationScene], None] | None,
    active_error: BaseException | None,
) -> None:
    cleanup_errors: list[BaseException] = []

    if finalize_func is not None:
        try:
            finalize_func(scene)
        except BaseException as error:
            cleanup_errors.append(error)

    try:
        plt.close(scene.figure)
    except BaseException as error:
        cleanup_errors.append(error)

    if not cleanup_errors:
        return

    if active_error is None:
        raise cleanup_errors[0]

    # for error in cleanup_errors: # only runs on remote (__version__ >= 3.11)
    #     active_error.add_note(
    #         "Additional error during animation cleanup: "
    #         f"{type(error).__name__}: {error}"
    #     )


def render_animation_chunk(
    job: AnimationJob[FrameT], 
    chunk: RenderChunk[FrameT], 
    progress: queue.Queue | _LocalProgress
) -> tuple[int, Path]:
    
    scene: AnimationScene | None = None
    active_error: BaseException | None = None

    try:
        scene = job.init_func()

        with job.writer.open_chunk(chunk.output_path, scene) as chunk_writer:
            for frame in chunk.frames:
                job.func(frame, scene)
                chunk_writer.write_frame(scene)
                progress.put(1)

    except BaseException as error:
        active_error = error
        raise

    finally:
        if scene is not None:
            finalize_animation_scene(scene, job.finalize_func, active_error)

    return chunk.index, chunk.output_path


def wait_for_workers(
    futures: Sequence[Future[tuple[int, Path]]],
    progress_queue: queue.Queue,
    total_frames: int,
) -> list[tuple[int, Path]]:
    
    pending = set(futures)
    completed_frames = 0
    results: list[tuple[int, Path]] = []
    started_at = time.perf_counter()

    try:
        while pending:
            try:
                completed_frames += progress_queue.get(timeout=0.1)
                elapsed = time.perf_counter() - started_at
                print_status(completed_frames, total_frames, elapsed)

            except queue.Empty:
                pass

            finished = { future for future in pending if future.done() }

            for future in finished:
                # Worker exceptions and remote tracebacks propagate here.
                results.append(future.result())

            pending.difference_update(finished)

        # Drain messages that reached the queue just after the futures
        # completed.
        while True:
            try:
                completed_frames += progress_queue.get_nowait()
            except queue.Empty:
                break

        elapsed=time.perf_counter() - started_at
        print_status(completed_frames, total_frames, elapsed)

    finally:
        print()

    return sorted( results, key=lambda result: result[0])
