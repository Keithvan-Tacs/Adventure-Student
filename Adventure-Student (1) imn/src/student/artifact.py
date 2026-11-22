"""Initial artifact data: an id, a position, and a reward value (no points logic here)."""
from src.api import ArtifactDefinition, Position


class Artifact:
    """Expose read-only canonical identity, position, and nonnegative value."""

    def __init__(self, artifact_id: str, position: Position, value: int) -> None:
        # ArtifactDefinition checks everything (str id, real Position, int >= 0).
        checked = ArtifactDefinition(artifact_id, position, value)
        self._artifact_id = checked.artifact_id
        # Make our own Position copy so the caller can't sneak in a different object.
        self._position = Position(checked.position.row, checked.position.column)
        self._value = checked.value

    @property
    def artifact_id(self) -> str:
        return self._artifact_id

    @property
    def position(self) -> Position:
        return self._position

    @property
    def value(self) -> int:
        return self._value