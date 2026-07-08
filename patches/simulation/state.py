"""Runtime game state and placement/step results."""

from __future__ import annotations

from dataclasses import dataclass, field

from .utils import Coordinate, Rect


@dataclass
class GameState:
    """Mutable Patches game state.

    ``patches`` maps a clue id to its currently committed rectangle.
    ``assignment`` maps each covered cell to the clue id covering it and is kept
    in sync with ``patches``. ``history`` stores snapshots of ``patches`` for
    undo.
    """

    rows: int
    cols: int
    patches: dict[int, Rect] = field(default_factory=dict)
    assignment: dict[Coordinate, int] = field(default_factory=dict)
    history: list[dict[int, Rect]] = field(default_factory=list, repr=False, compare=False)
    done: bool = False
    success: bool = False
    step_count: int = 0

    @property
    def covered_count(self) -> int:
        return len(self.assignment)

    def clone(self) -> "GameState":
        return GameState(
            rows=self.rows,
            cols=self.cols,
            patches=dict(self.patches),
            assignment=dict(self.assignment),
            history=[dict(snapshot) for snapshot in self.history],
            done=self.done,
            success=self.success,
            step_count=self.step_count,
        )


@dataclass(frozen=True)
class StepResult:
    state: GameState
    valid: bool
    reason: str | None
    solved: bool
