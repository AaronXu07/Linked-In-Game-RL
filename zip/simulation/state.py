"""Runtime game state and step results."""

from __future__ import annotations

from dataclasses import dataclass, field

from .utils import Coordinate


@dataclass
class GameState:
    path: list[Coordinate]
    visited: set[Coordinate]
    current_pos: Coordinate
    next_waypoint_index: int
    done: bool
    success: bool
    step_count: int
    waypoint_history: list[int] = field(default_factory=list, repr=False, compare=False)

    def clone(self) -> "GameState":
        return GameState(
            path=list(self.path),
            visited=set(self.visited),
            current_pos=self.current_pos,
            next_waypoint_index=self.next_waypoint_index,
            done=self.done,
            success=self.success,
            step_count=self.step_count,
            waypoint_history=list(self.waypoint_history),
        )


@dataclass(frozen=True)
class StepResult:
    state: GameState
    valid: bool
    reason: str | None
    solved: bool
