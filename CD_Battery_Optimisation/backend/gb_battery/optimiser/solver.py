"""Solver abstraction.

HiGHS (via ``highspy`` through Pyomo's APPSI interface) is the default open-source
solver. A Gurobi adapter is offered *only* when a licence is detected — it is never
mandatory.
"""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass

import pyomo.environ as pyo

# APPSI/HiGHS captures OS-level file descriptors while solving. In a server worker
# thread (e.g. FastAPI's threadpool) that capture deadlocks unless fds 1/2 point at
# real files. We therefore redirect stdout/stderr to os.devnull for the duration of
# the solve and serialise solves with a process-wide lock (fd manipulation is global).
_SOLVE_LOCK = threading.Lock()


@contextlib.contextmanager
def _fd_guard():
    with _SOLVE_LOCK:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved_out, saved_err = os.dup(1), os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            yield
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(devnull)
            os.close(saved_out)
            os.close(saved_err)


def guarded_solve(opt, model: pyo.ConcreteModel):
    """Run ``opt.solve(model)`` with fd redirection + a global lock (server-safe)."""
    with _fd_guard():
        return opt.solve(model)


class SolverNotAvailable(RuntimeError):
    """Raised when no supported solver can be constructed."""


@dataclass
class SolveOutcome:
    status: str  # "optimal" | "infeasible" | "unbounded" | "error"
    solver: str
    objective: float | None


def _try_gurobi():  # pragma: no cover - depends on optional licence
    try:
        from pyomo.contrib.appsi.solvers.gurobi import Gurobi

        opt = Gurobi()
        if not opt.available():
            return None
        return opt
    except Exception:
        return None


def _highs():
    from pyomo.contrib.appsi.solvers.highs import Highs

    return Highs()


def get_solver(prefer_gurobi: bool = False):
    """Return an APPSI solver instance (HiGHS by default)."""
    if prefer_gurobi:
        g = _try_gurobi()
        if g is not None:
            return g, "gurobi"
    try:
        return _highs(), "highs"
    except Exception as exc:  # pragma: no cover
        raise SolverNotAvailable(f"HiGHS unavailable: {exc}") from exc


_TERMINATION = {
    "optimal": "optimal",
    "maxTimeLimit": "time_limit",
    "maxIterations": "iteration_limit",
    "infeasible": "infeasible",
    "unbounded": "unbounded",
    "infeasibleOrUnbounded": "infeasible",
}


def solve(model: pyo.ConcreteModel, prefer_gurobi: bool = False) -> SolveOutcome:
    """Solve ``model`` and return a normalised outcome."""
    opt, name = get_solver(prefer_gurobi=prefer_gurobi)
    opt.config.stream_solver = False
    opt.config.load_solution = False
    results = guarded_solve(opt, model)

    cond = str(results.termination_condition)
    # APPSI enums stringify like "TerminationCondition.optimal".
    key = cond.split(".")[-1]
    status = _TERMINATION.get(key, key)

    if status in {"optimal", "time_limit", "iteration_limit"}:
        results.solution_loader.load_vars()
        obj = float(pyo.value(model.objective))
        return SolveOutcome(status="optimal" if status == "optimal" else status,
                            solver=name, objective=obj)
    return SolveOutcome(status=status, solver=name, objective=None)
