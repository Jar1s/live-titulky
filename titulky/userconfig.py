"""Užívateľská konfigurácia — UI Settings.

Ukladá sa do ~/.config/titulky/config.json. Aplikuje sa do os.environ
pri štarte PRED initom pipeline; reálna env premenná má prednosť
(power-user override), hodnota z configu sa použije, len ak env chýba.
"""
import json
import os
import pathlib

PATH = pathlib.Path.home() / ".config" / "titulky" / "config.json"

# polia editovateľné z UI; order = poradie vo formulári
FIELDS = [
    ("OPENAI_API_KEY",      "secret", "OpenAI API kľúč (cloud režim)"),
    ("AZURE_SPEECH_KEY",    "secret", "Azure Speech kľúč (korekcia, voliteľné)"),
    ("AZURE_SPEECH_REGION", "text",   "Azure región (napr. swedencentral)"),
    ("TITULKY_ENGINE",      "select:local,cloud", "Predvolený engine"),
    ("TITULKY_CLOUD_MODEL", "select:gpt-live-transcribe,gpt-4o-transcribe,"
                            "gpt-4o-mini-transcribe", "Cloud model"),
    ("TITULKY_DEVICE",      "device", "Mikrofón (prázdne = systémový)"),
    ("TITULKY_LANGUAGE",    "text",   "Jazyk (sk)"),
    ("TITULKY_SPELLFIX",    "bool",   "Spellfix lexikón (doménové prednášky)"),
    ("TITULKY_NO_AZURE",    "bool",   "Vypnúť Azure korekciu (plný offline)"),
    ("TITULKY_FINAL_REPO",  "text",   "Vlastný final model (HF repo/cesta)"),
]
SECRET_KEYS = {k for k, t, _ in FIELDS if t == "secret"}
KEYS = [k for k, _, _ in FIELDS]


def load() -> dict:
    try:
        return json.loads(PATH.read_text())
    except Exception:
        return {}


def save(cfg: dict):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if k in KEYS and v not in ("", None)}
    PATH.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
    try:
        PATH.chmod(0o600)   # kľúče — len vlastník
    except Exception:
        pass
    return clean


def apply():
    """Konfig -> os.environ; existujúca env premenná vyhráva."""
    for k, v in load().items():
        if k in KEYS and not os.environ.get(k):
            os.environ[k] = str(v)


def masked() -> dict:
    """Konfig pre UI — tajomstvá maskované na posledné 4 znaky."""
    out = {}
    for k, v in load().items():
        if k in SECRET_KEYS and v:
            out[k] = "••••" + str(v)[-4:]
        else:
            out[k] = v
    return out


def merge_from_ui(incoming: dict) -> dict:
    """Ulož z UI: maskované tajomstvo (začína ••••) = ponechaj staré."""
    cur = load()
    for k, v in incoming.items():
        if k not in KEYS:
            continue
        if k in SECRET_KEYS and isinstance(v, str) and v.startswith("••••"):
            continue            # nezmenené — drž pôvodné
        if v in ("", None, False):
            cur.pop(k, None)
        else:
            cur[k] = v
    return save(cur)


def field_meta():
    return [{"key": k, "type": t, "label": lb} for k, t, lb in FIELDS]
