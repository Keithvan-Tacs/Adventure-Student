"""The explorer's identity. It only keeps the agent_id; it does not store position."""
from src.api import CharacterDefinition


class Character:
    """Character(agent_id); exposes a read-only str agent_id property."""

    def __init__(self, agent_id: str) -> None:
        # CharacterDefinition already checks: must be a str, not empty, no extra spaces.
        # We keep our own copy so nobody can change it later.
        self._agent_id = CharacterDefinition(agent_id).agent_id

    @property
    def agent_id(self) -> str:
        return self._agent_id