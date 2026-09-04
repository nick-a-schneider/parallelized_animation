from dataclasses import dataclass
from typing import Any

from matplotlib.artist import Artist
from matplotlib.figure import Figure

@dataclass
class AnimationScene:
    figure: Figure
    artists: dict[str, Artist]
    state: Any = None
