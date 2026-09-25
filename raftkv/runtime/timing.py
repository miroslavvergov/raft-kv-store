"""A node's clock settings: how long a tick is, and how many ticks each timeout lasts."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Timing:
    """The tick length and the timeouts counted in ticks (DD-9).

    Time inside a node is a count of ticks; only the clock that ticks it knows
    how long one lasts. The defaults are a 100 ms tick, a heartbeat every tick,
    and an election timeout drawn from 10 to 19 ticks.

    Attributes:
        tick_interval: Seconds between two ticks of a running node.
        heartbeat_ticks: Ticks between two heartbeats of a Leader (REPL-9).
        election_ticks: The shortest election timeout, in ticks; each timeout is
            drawn from `election_ticks` to `2 * election_ticks - 1` (ELECT-1).

    Raises:
        ValueError: If `tick_interval` is not positive, `heartbeat_ticks` is below
            1, or `election_ticks` is not above `heartbeat_ticks`, which REPL-10
            requires so a heartbeat always comes before a timeout can fire.
    """

    tick_interval: float = 0.1
    heartbeat_ticks: int = 1
    election_ticks: int = 10

    def __post_init__(self) -> None:
        if self.tick_interval <= 0:
            raise ValueError(f"tick_interval must be positive, got {self.tick_interval}")
        if self.heartbeat_ticks < 1:
            raise ValueError(f"heartbeat_ticks must be at least 1, got {self.heartbeat_ticks}")
        if self.election_ticks <= self.heartbeat_ticks:
            raise ValueError(
                f"election_ticks ({self.election_ticks}) must be above "
                f"heartbeat_ticks ({self.heartbeat_ticks})"
            )

    def random_election_timeout(self, rng: random.Random) -> int:
        """Return a new election timeout, in ticks, from `election_ticks` to twice it, exclusive.

        Args:
            rng: The source of randomness; a seeded one draws the same timeouts again.
        """
        return rng.randrange(self.election_ticks, 2 * self.election_ticks)
