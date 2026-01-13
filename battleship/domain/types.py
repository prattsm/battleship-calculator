from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Placement:
    ship: str
    cells: Tuple[Tuple[int, int], ...]
    mask: int
