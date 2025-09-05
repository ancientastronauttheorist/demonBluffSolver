"""Role registry and base classes."""

from __future__ import annotations

from typing import Dict, Type


class Role:
    """Base role adapter."""
    name: str = "role"

    @staticmethod
    def static_constraints(world, puzzle):  # pragma: no cover - stubs
        return []

    def simulate_action(self, world, action_args):  # pragma: no cover - stubs
        return []


ROLE_REGISTRY: Dict[str, Type[Role]] = {}


def register_role(cls: Type[Role]) -> Type[Role]:
    ROLE_REGISTRY[cls.name] = cls
    return cls
