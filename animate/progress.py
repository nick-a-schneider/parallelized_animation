import time

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def print_status(completed: int, total_frames: int, elapsed: float) -> None:
    percent = 100.0 * completed / total_frames
    rate = completed / elapsed if elapsed > 0.0 else 0.0

    print(
        f"\rRendering frame {completed}/{total_frames} "
        f"({percent:5.1f}%) | "
        f"elapsed {format_duration(elapsed)} | "
        f"{rate:5.2f} frames/s",
        end="",
        flush=True,
    )
    
class _LocalProgress:
    def __init__(self, total_frames: int) -> None:
        self.total_frames = total_frames
        self.completed = 0
        self.started_at = time.perf_counter()

    def put(self, amount: int) -> None:
        self.completed += amount
        elapsed = time.perf_counter() - self.started_at
        print_status(self.completed, self.total_frames, elapsed)
        if self.completed >= self.total_frames:
            print()
