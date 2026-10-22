"""Slovníková korekčná vrstva — deterministické opravy známych chýb ASR.

Cieľ: anglicizmy a domáce výrazy, ktoré SK fine-tune komolí foneticky
(„hartéchno" → „hard techno", „afromelody" → „Afromelodic").

Zámerne jednoduchá: presná zhoda tokenu/bigramu po normalizácii (lowercase,
bez diakritiky, bez interpunkcie na okrajoch). Žiadne fuzzy — fuzzy robí
Azure vrstva s gatmi; slovník musí byť 100 % predvídateľný.

Súbor `lexikon.json` v koreni repa: {"zly tvar": "správny tvar", ...}
— kľúč môže byť 1 alebo 2 slová; hodnota ľubovoľný počet slov.
Needituje sa kód, len JSON.
"""
import json
import os
import unicodedata

_DEFAULT = {
    # merané na podcaste Jara (BENCHMARK.md §3.7)
    "hartéchno": "hard techno",
    "hartechno": "hard techno",
    "hart techno": "hard techno",
    "afromelody": "Afromelodic",
    "afromelodik": "Afromelodic",
}


def _norm(t: str) -> str:
    t = t.strip().strip(".,!?…:;\"'").lower()
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


class Lexicon:
    def __init__(self, path: str | None = None):
        data = dict(_DEFAULT)
        path = path or os.path.join(os.path.dirname(__file__), "..",
                                    "lexikon.json")
        try:
            data.update(json.load(open(path)))
        except (OSError, json.JSONDecodeError):
            pass  # bez súboru beží default
        self.uni = {}   # 1-slovný kľúč
        self.bi = {}    # 2-slovný kľúč
        for k, v in data.items():
            parts = k.split()
            if len(parts) == 2:
                self.bi[(_norm(parts[0]), _norm(parts[1]))] = v
            elif len(parts) == 1:
                self.uni[_norm(k)] = v

    def apply(self, words: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        """Vráti (opravené slová, [(z, na), ...]). Interpunkciu okrajov drží."""
        out = []
        edits = []
        i = 0
        while i < len(words):
            w = words[i]
            key = _norm(w)
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt is not None and (key, _norm(nxt)) in self.bi:
                rep = self.bi[(key, _norm(nxt))]
                tail = nxt[len(nxt.rstrip(".,!?…:;")):]  # interpunkcia konca
                out.extend((rep + tail).split())
                edits.append((f"{w} {nxt}", rep))
                i += 2
                continue
            if key in self.uni:
                rep = self.uni[key]
                head = w[:len(w) - len(w.lstrip())]
                tail = w[len(w.rstrip(".,!?…:;")):]
                out.extend((head + rep + tail).split())
                edits.append((w, rep))
                i += 1
                continue
            out.append(w)
            i += 1
        return out, edits
