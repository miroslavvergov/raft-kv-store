# raft-kv-store

A small distributed key-value store replicated with a from-scratch
implementation of the Raft consensus algorithm (Ongaro & Ousterhout, 2014),
written in Python 3 with asyncio.

Full specification: see the accompanying A4 Software Requirements
Specification for the complete requirements, design decisions, and test
strategy this project implements against.

## Development setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pytest -q

## Status

Environment and repository scaffold only — implementation has not started yet.
