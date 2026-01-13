from dataclasses import dataclass
from typing import Dict, List

from .definition import LayoutDefinition
from .placements import LayoutPlacement, generate_layout_placements


@dataclass
class LayoutRuntime:
    definition: LayoutDefinition
    placements: Dict[str, List[LayoutPlacement]]


class PlacementCache:
    def __init__(self):
        self._cache: Dict[str, LayoutRuntime] = {}

    def _key(self, layout: LayoutDefinition) -> str:
        return f"{layout.layout_hash}:{layout.layout_version}"

    def get(self, layout: LayoutDefinition) -> LayoutRuntime:
        key = self._key(layout)
        if key not in self._cache:
            placements = generate_layout_placements(layout)
            self._cache[key] = LayoutRuntime(layout, placements)
        return self._cache[key]


DEFAULT_PLACEMENT_CACHE = PlacementCache()
