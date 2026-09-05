from __future__ import annotations

import time
import queue

from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Sequence, Protocol

import matplotlib.pyplot as plt

##################################################
from .scene import AnimationScene
from .progress import _LocalProgress, print_status
from .types import FrameT, RenderChunk, AnimationJob, ProgressSink, ProgressQueue

        
def finalize_animation_scene(
    scene: AnimationScene,
    finalize_func: Callable[[AnimationScene], None] | None,
    active_error: BaseException | None,
) -> None:
    """
    Finalize and close an animation scene while preserving active failures.

    Runs the optional user finalizer and closes the Matplotlib figure. Cleanup
    failures are raised only when no earlier rendering error is already active.

    Parameters
    ----------
    scene : AnimationScene
        Scene to finalize and close.
    finalize_func : Callable[[AnimationScene], None] | None
        Optional user cleanup callback.
    active_error : BaseException | None
        Exception already being propagated from rendering, if any.

    Raises
    ------
    BaseException
        If cleanup fails and no earlier exception is active.
    """
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
    progress: ProgressSink,
) -> int:
    """
    Render one contiguous frame chunk to independent PNG files.

    Creates one worker-local scene, renders each assigned frame to a
    deterministically named PNG file, reports progress after each frame,
    then finalizes the scene.

    Parameters
    ----------
    job : AnimationJob[FrameT]
        Scene initialization, update, and finalization callbacks.
    chunk : RenderChunk[FrameT]
        Contiguous subset of frames assigned to this task.
    progress : ProgressSink
        Progress sink receiving one completion event per frame.

    Returns
    -------
    int
        Index of the completed render chunk.

    Raises
    ------
    BaseException
        Propagates initialization, rendering, writing, or cleanup failures.
    """
    
    scene: AnimationScene | None = None
    active_error: BaseException | None = None

    try:
        scene = job.init_func()

        for offset, frame in enumerate(chunk.frames):
            frame_index = chunk.start + offset
            frame_path = chunk.output_directory / f"frame_{frame_index:08d}.png"

            job.func(frame, scene)

            scene.figure.savefig(frame_path, format="png", dpi=job.config.dpi)

            progress.put(1)

    except BaseException as error:
        active_error = error
        raise

    finally:
        if scene is not None:
            finalize_animation_scene(scene, job.finalize_func, active_error)

    return chunk.index


def wait_for_workers(
    futures: Sequence[Future[int]],
    progress_queue: ProgressQueue,
    total_frames: int,
) -> list[int]:
    """
    Wait for worker completion while displaying aggregate render progress.

    Polls progress events and completed futures until all workers finish,
    propagating worker exceptions and returning completed chunk indices.

    Parameters
    ----------
    futures : Sequence[Future[int]]
        Worker futures producing completed chunk indices.
    progress_queue : ProgressQueue
        Queue receiving completed-frame counts from workers.
    total_frames : int
        Total number of frames across all chunks.

    Returns
    -------
    list[int]
        Completed chunk indices in ascending order.

    Raises
    ------
    BaseException
        Propagates exceptions raised by worker futures.
    """
    
    pending = set(futures)
    completed_frames = 0
    results: list[int] = []
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

    return sorted(results)