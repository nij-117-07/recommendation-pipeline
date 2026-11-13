"""Minimal DAG executor: runs tasks in dependency order, retries failures, and
skips every task downstream of a failure. Every step is logged to a run log."""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from src.common.config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Task:
    name: str
    run: Callable[[dict], None]
    deps: tuple = field(default=())
    retries: int = 0


def topological_order(tasks):
    order, state = [], {}

    def visit(name, path):
        if name not in tasks:
            raise KeyError(f"Unknown dependency: {name}")
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            raise ValueError(f"Cycle detected: {' -> '.join(path + [name])}")
        state[name] = "visiting"
        for dep in tasks[name].deps:
            visit(dep, path + [name])
        state[name] = "done"
        order.append(name)

    for name in tasks:
        visit(name, [])
    return order


def run_workflow(tasks, cfg, log_file=None):
    handler = None
    if log_file is not None:
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        logging.getLogger().addHandler(handler)

    try:
        order = topological_order(tasks)
        log.info("Execution order: %s", " -> ".join(order))
        status = {}
        for name in order:
            task = tasks[name]
            blocked = [d for d in task.deps if status.get(d) != "success"]
            if blocked:
                status[name] = "skipped"
                log.warning("[%s] skipped because of failed dependencies: %s", name, blocked)
                continue

            for attempt in range(task.retries + 1):
                log.info("[%s] started (attempt %d)", name, attempt + 1)
                started = time.perf_counter()
                try:
                    task.run(cfg)
                except Exception:
                    log.exception("[%s] failed", name)
                    status[name] = "failed"
                else:
                    status[name] = "success"
                    log.info("[%s] success in %.2fs", name, time.perf_counter() - started)
                    break

        log.info("Workflow summary: %s", status)
        if any(s != "success" for s in status.values()):
            raise RuntimeError(f"Workflow finished with problems: {status}")
        return status
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
