"""Run-trace writer — zápis behov do .runs/<run>/trace.jsonl (RUNTRACE-SPEC v1.2).

Minimálna, stdlib-only implementácia pre live titulky. Viewer: Projektovník
BUILD tab Flow (číta .runs/ každého repa v ~/OS/ACTUAL) a Hermes build-pane.
Kanonika: projektovnik/docs/RUNTRACE-SPEC.md, vzor grindcast_auto/trace.py.
"""
from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import threading
import time
from datetime import UTC, datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lock = threading.Lock()


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(["git", "-C", BASE, *args], capture_output=True,
                             text=True, timeout=5, check=False).stdout.strip()
        return out or None
    except OSError:
        return None


class RunTrace:
    """Jeden beh = jedna inštancia. Thread-safe append do trace.jsonl."""

    def __init__(self, nazov: str, spustil: str = "rucne"):
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
        self.run = f"{nazov}-{ts}-{secrets.token_hex(2)}"
        self.dir = os.path.join(BASE, ".runs", self.run)
        os.makedirs(self.dir, exist_ok=True)
        self.trace = os.path.join(self.dir, "trace.jsonl")
        self._seq = 0
        t = time.time()
        self._zapis({
            "run": self.run, "seq": self._next(), "krok": "__run_start__",
            "t0": t, "t1": t, "stav": "ok",
            "meta": {"commit": _git(["rev-parse", "--short", "HEAD"]),
                     "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
                     "spustil": spustil, "host": socket.gethostname()},
        })

    def _next(self) -> int:
        with _lock:
            s = self._seq
            self._seq += 1
            return s

    def _zapis(self, riadok: dict):
        with _lock, open(self.trace, "a", encoding="utf-8") as f:
            f.write(json.dumps(riadok, ensure_ascii=False) + "\n")

    def krok(self, nazov: str, t0: float, t1: float | None = None,
             stav: str = "ok", vstupy: list | None = None,
             vystup: dict | None = None, chyba: dict | None = None,
             meta: dict | None = None, parent: int | None = None) -> int:
        """Zapíš hotový krok (jeden finálny riadok; viewer berie posledný
        riadok per seq, takže 'beží' medzistav pri krátkych krokoch netreba)."""
        seq = self._next()
        self._zapis({
            "run": self.run, "seq": seq, "krok": nazov,
            "t0": t0, "t1": t1 if t1 is not None else time.time(),
            "stav": stav, "vstupy": vstupy or [], "vystup": vystup,
            "chyba": chyba, "meta": meta or {}, "parent": parent,
        })
        return seq


def vstup(nazov: str, typ: str = "data", pocet: int | None = None,
          zdroj: str = "", ukazka: str = "") -> dict:
    return {"nazov": nazov, "typ": typ, "pocet": pocet,
            "zdroj": zdroj, "ukazka": ukazka[:200]}


def vystup(typ: str, ukazka: str = "", pocet: int | None = None, **extra) -> dict:
    return {"typ": typ, "pocet": pocet, "ukazka": ukazka[:400], **extra}
