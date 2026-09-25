"""Tier 1 tests for Timing: defaults, REPL-10's ordering, and ELECT-1's random timeouts (DD-9)."""

import random

import pytest

from raftkv.runtime import Timing


def test_the_defaults_are_a_100_ms_tick_a_heartbeat_every_tick_and_10_to_19_tick_timeouts():
    assert Timing() == Timing(tick_interval=0.1, heartbeat_ticks=1, election_ticks=10)


@pytest.mark.parametrize(
    ("options", "why"),
    [
        ({"tick_interval": 0}, "a tick must last some time"),
        ({"heartbeat_ticks": 0}, "a heartbeat interval must be at least one tick"),
        ({"heartbeat_ticks": 5, "election_ticks": 5}, "REPL-10: a heartbeat equal to a timeout"),
        ({"heartbeat_ticks": 6, "election_ticks": 5}, "REPL-10: a heartbeat above a timeout"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_settings_that_break_a_rule_are_refused(options, why):
    with pytest.raises(ValueError):
        Timing(**options)


def test_an_election_timeout_one_tick_above_the_heartbeat_is_accepted():
    assert Timing(heartbeat_ticks=5, election_ticks=6).election_ticks == 6


def test_every_election_timeout_is_from_the_shortest_to_just_under_twice_it():
    timing, rng = Timing(election_ticks=10), random.Random(0)
    drawn = {timing.random_election_timeout(rng) for _ in range(500)}
    assert drawn == set(range(10, 20))


def test_the_same_seed_draws_the_same_timeouts():
    timing = Timing()
    first, second = random.Random(7), random.Random(7)
    assert [timing.random_election_timeout(first) for _ in range(20)] == [
        timing.random_election_timeout(second) for _ in range(20)
    ]
