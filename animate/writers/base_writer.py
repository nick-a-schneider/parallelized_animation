from __future__ import annotations

import shutil

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType

from ..scene import AnimationScene

class AnimationChunkWriter(ABC):
    """
    Worker-local destination for one contiguous animation chunk.

    Implementations may own an FFmpeg process, PdfPages instance,
    image sequence, or another output resource.
    """

    def __enter__(self) -> AnimationChunkWriter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exception is None:
            self.finish()
        else:
            self.abort()

        return False

    @abstractmethod
    def write_frame(self, scene: AnimationScene) -> None:
        """Write the scene's current state as the next output frame."""

    @abstractmethod
    def finish(self) -> None:
        """
        Complete the chunk and validate its output.

        Any encoder or output failure should propagate from this method.
        """

    @abstractmethod
    def abort(self) -> None:
        """
        Stop writing and release resources after a rendering failure.

        This method should avoid concealing the active rendering exception.
        """

class AnimationWriter(ABC):
    """
    Configured output-format writer.

    Instances are serialized and sent to animation workers, so persistent
    configuration must be pickleable. Mutable per-chunk resources belong in
    AnimationChunkWriter instances.
    """
    supports_parallel: bool = False

    @property
    @abstractmethod
    def chunk_suffix(self) -> str:
        """Filename suffix used for worker-local chunk outputs."""

    @abstractmethod
    def open_chunk(self, output_path: Path, scene: AnimationScene) -> AnimationChunkWriter:
        """
        Create a worker-local writer for one animation chunk.

        The returned writer may inspect or prepare the scene before frame
        updates begin.
        """

    def combine_chunks(self, chunk_paths: list[Path], output_path: Path, temporary_directory: Path) -> None:
        """
        Combine completed chunks in their given order into the final output.

        This runs in the calling process after every worker succeeds.
        """
        del temporary_directory
        numchunks = len(chunk_paths)
        if numchunks != 1:
            raise ValueError(
                f"{type(self).__name__} does not support "
                f"combining multiple chunks [{numchunks}]"
            )

        shutil.move(chunk_paths[0], output_path)