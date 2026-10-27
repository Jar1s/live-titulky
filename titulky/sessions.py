"""História sessionov — ukladanie prepisov a metrík na disk.

Jeden JSON na session v ~/.config/titulky/sessions/. Session sa uloží
pri prepnutí enginu, pri ukončení servera a autosave každých 30 s
(crash nestratí viac než pol minúty).
"""
import json
import os
import pathlib
import statistics
import time

DIR = pathlib.Path.home() / ".config" / "titulky" / "sessions"


def new(engine: str | None, model: str | None) -> dict:
    """Nový session stav (drží ho main.py)."""
    return {"id": time.strftime("%Y%m%d-%H%M%S"), "t0": time.time(),
            "engine": engine or "local", "model": model}


def _p50(xs):
    return round(statistics.median(xs), 2) if xs else None


def save(pipe, sess: dict) -> bool:
    """Ulož session; prepisuje ten istý súbor (autosave). False = nič na uloženie."""
    if pipe is None or sess is None or not sess.get("id"):
        return False
    text = " ".join(getattr(pipe, "final_text", []) or [])
    words = len(text.split())
    if words == 0:
        return False
    flicker = None
    agree = getattr(pipe, "final_agree", None)
    if agree is not None:
        try:
            flicker = round(agree.flicker_rate(), 3)
        except Exception:
            flicker = None
    data = {
        "id": sess["id"],
        "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(sess["t0"])),
        "duration_s": round(time.time() - sess["t0"], 1),
        "engine": sess.get("engine"),
        "model": sess.get("model"),
        "words": words,
        "draft_p50_s": _p50(getattr(pipe, "draft_latencies", None) or []),
        "final_p50_s": _p50(getattr(pipe, "final_latencies", None) or []),
        "flicker": flicker,
        "transcript": text,
    }
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = DIR / f".{sess['id']}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(DIR / f"{sess['id']}.json")
    return True


def list_all() -> list[dict]:
    """Súhrny všetkých sessionov, najnovšie prvé (bez prepisu)."""
    out = []
    if not DIR.is_dir():
        return out
    for p in sorted(DIR.glob("*.json"), reverse=True):
        try:
            d = json.loads(p.read_text())
            d["preview"] = d.get("transcript", "")[:140]
            d.pop("transcript", None)
            out.append(d)
        except Exception:
            continue
    return out


def get(sid: str) -> dict | None:
    p = DIR / f"{os.path.basename(sid)}.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def delete(sid: str) -> bool:
    p = DIR / f"{os.path.basename(sid)}.json"
    try:
        p.unlink()
        return True
    except OSError:
        return False
