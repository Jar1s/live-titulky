"""Vstupný bod: spustí WebSocket server + dvojvrstvovú pipeline.

Použitie:
  python -m titulky.main                # živý mikrofón
  python -m titulky.main --file x.wav   # simulovaný live vstup zo súboru
Overlay: otvor http://localhost:8765 v prehliadači.
"""
import argparse
import asyncio
import json
import os
import pathlib
import statistics
import threading
import time

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

from . import trace as rt
from .pipeline import TwoTierPipeline

CLIENTS = set()
LOOP = None
PIPE = None   # beziaca pipeline — pre UI pause toggle
ENGINE = {"engine": None, "model": None, "switching": False,
          "language": "sk"}
ENGINE_LOCK = threading.Lock()
AUDIO_DIR = None  # nastaví main() po vytvorení trace adresára
HTML = (pathlib.Path(__file__).parent / "overlay.html").read_bytes()
FLOW = (pathlib.Path(__file__).parent / "flow.html").read_bytes()
STUDIO = (pathlib.Path(__file__).parent / "studio.html").read_bytes()
LAB = (pathlib.Path(__file__).parent / "lab.html").read_bytes()

# modely dostupné v Studio prepínači
ENGINE_CATALOG = [
    {"id": "local", "label": "Lokál — kinit-turbo-sk (offline)",
     "engine": "local", "model": None},
    {"id": "cloud-live", "label": "Cloud — gpt-live-transcribe",
     "engine": "cloud", "model": "gpt-live-transcribe"},
    {"id": "cloud-4o", "label": "Cloud — gpt-4o-transcribe",
     "engine": "cloud", "model": "gpt-4o-transcribe"},
    {"id": "cloud-mini", "label": "Cloud — gpt-4o-mini-transcribe",
     "engine": "cloud", "model": "gpt-4o-mini-transcribe"},
]


def _on_update(d):
    broadcast(d)


def _engine_state_msg():
    return {"event": {"t": "engine_state", **{k: ENGINE[k] for k in
                                              ("engine", "model",
                                               "switching")},
                      "catalog": ENGINE_CATALOG}}


def switch_engine(engine: str, model: str | None):
    """Zastav bežiacu pipeline a naštartuj novú (mikrofón)."""
    global PIPE
    with ENGINE_LOCK:
        if ENGINE["switching"]:
            return
        ENGINE["switching"] = True
    broadcast(_engine_state_msg())
    try:
        old = PIPE
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass
        if engine == "cloud":
            from .cloud_engine import CloudPipeline
            if model:
                from . import cloud_engine
                cloud_engine.CLOUD_MODEL = model
            pipe = CloudPipeline(language=ENGINE["language"],
                                 on_update=_on_update,
                                 on_event=lambda e: broadcast({"event": e}))
            pipe.start_mic(os.environ.get("TITULKY_DEVICE") or None)
        else:
            pipe = TwoTierPipeline(language=ENGINE["language"],
                                   on_update=_on_update,
                                   on_event=lambda e: broadcast(
                                       {"event": e}))
            pipe.warmup()
            pipe.start_mic()
            threading.Thread(target=pipe.run, daemon=True).start()
        PIPE = pipe
        ENGINE.update(engine=engine, model=model)
    except Exception as e:
        broadcast({"event": {"t": "cloud_err",
                             "msg": f"switch zlyhal: {e}"[:200]}})
    finally:
        ENGINE["switching"] = False
        broadcast(_engine_state_msg())


def broadcast(data: dict):
    if LOOP is None:
        return
    msg = json.dumps(data, ensure_ascii=False)
    for ws in list(CLIENTS):
        asyncio.run_coroutine_threadsafe(ws.send(msg), LOOP)


async def handler(ws):
    CLIENTS.add(ws)
    try:
        if PIPE is not None and PIPE.paused:
            await ws.send(json.dumps({"event": {"t": "paused", "on": True}}))
        async for raw in ws:
            try:
                d = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if d.get("t") == "pause" and PIPE is not None:
                PIPE.paused = not PIPE.paused
                broadcast({"event": {"t": "paused", "on": PIPE.paused}})
            elif d.get("t") == "engine":
                threading.Thread(target=switch_engine,
                                 args=(d.get("engine", "local"),
                                       d.get("model")),
                                 daemon=True).start()
            elif d.get("t") == "state":
                await ws.send(json.dumps(_engine_state_msg(),
                                         ensure_ascii=False))
            elif d.get("t") == "transcript" and PIPE is not None:
                await ws.send(json.dumps(
                    {"event": {"t": "transcript",
                               "text": " ".join(PIPE.final_text)}},
                    ensure_ascii=False))
    finally:
        CLIENTS.discard(ws)


def serve_html(connection, request):
    if request.path == "/":
        return Response(200, "OK", Headers({"Content-Type": "text/html; charset=utf-8"}), HTML)
    if request.path == "/flow":
        return Response(200, "OK", Headers({"Content-Type": "text/html; charset=utf-8"}), FLOW)
    if request.path == "/studio":
        return Response(200, "OK", Headers({"Content-Type": "text/html; charset=utf-8"}), STUDIO)
    if request.path == "/lab":
        return Response(200, "OK", Headers({"Content-Type": "text/html; charset=utf-8"}), LAB)
    if request.path.startswith("/audio/") and AUDIO_DIR:
        # audit WAV segmentov — len basename, žiadne traverzovanie
        name = os.path.basename(request.path[len("/audio/"):])
        p = os.path.join(AUDIO_DIR, name)
        if name.endswith(".wav") and os.path.isfile(p):
            with open(p, "rb") as fh:
                return Response(200, "OK",
                                Headers({"Content-Type": "audio/wav"}),
                                fh.read())
        return Response(404, "Not Found", Headers({"Content-Type": "text/plain"}), b"not found")
    if request.path.startswith("/replay") and AUDIO_DIR:
        # A/B replay uloženého úseku cez viac modelov — výsledok príde
        # ako websocket event {t:'replay'}, nech handler neblokuje loop
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(request.path).query)
        name = os.path.basename(q.get("wav", [""])[0])
        if name.endswith(".wav") and os.path.isfile(os.path.join(AUDIO_DIR, name)):
            threading.Thread(target=_replay_worker, args=(name,),
                             daemon=True).start()
            return Response(200, "OK", Headers({"Content-Type": "application/json"}),
                            b'{"ok": true}')
        return Response(404, "Not Found", Headers({"Content-Type": "text/plain"}), b"no wav")
    return None  # ostatné cesty = WebSocket upgrade


_REPLAY_ASR = {}


def _replay_worker(name):
    """Prežene uložený WAV úseku viacerými modelmi a broadcastne porovnanie."""
    import wave as wv

    import numpy as np

    from .asr import ASR, DRAFT_REPO, FINAL_REPO
    try:
        with wv.open(os.path.join(AUDIO_DIR, name)) as wf:
            audio = (np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
                     .astype("float32") / 32768.0)
        rows = []
        for label, repo, beam in (
                ("draft (base)", DRAFT_REPO, 1),
                ("final (kinit)", FINAL_REPO, 5),
                ("turbo (vanilla)", "mlx-community/whisper-large-v3-turbo", 5)):
            key = (repo, beam)
            if key not in _REPLAY_ASR:
                _REPLAY_ASR[key] = ASR(repo=repo, beam=beam)
            t0 = time.monotonic()
            words = _REPLAY_ASR[key].transcribe(audio) or []
            rows.append({"model": label,
                         "text": " ".join(w["text"] for w in words),
                         "dec_s": round(time.monotonic() - t0, 2)})
        broadcast({"event": {"t": "replay", "wav": name, "rows": rows}})
    except Exception as ex:  # noqa: BLE001 — debug nástroj, chybu ukáž v UI
        broadcast({"event": {"t": "replay", "wav": name,
                             "rows": [{"model": "CHYBA", "text": str(ex),
                                       "dec_s": 0}]}})


async def ws_main(host, port):
    global LOOP
    LOOP = asyncio.get_running_loop()
    async with websockets.serve(handler, host, port, process_request=serve_html):
        print(f"Overlay: http://{host}:{port}")
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="WAV 16 kHz mono namiesto mikrofónu")
    ap.add_argument("--language", default="sk")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    global PIPE

    def on_update(d):
        print(f"[{d['audio_s']:6.1f}s draft={d['draft_latency_s'] and round(d['draft_latency_s'],2)} "
              f"final={d['final_latency_s'] and round(d['final_latency_s'],2)}] "
              f"{d['committed'][-70:]} | {d['pending'][:50]}")
        broadcast(d)

    if os.environ.get("TITULKY_ENGINE") == "cloud":
        # cloud režim: OpenAI Realtime STT namiesto lokálnej pipeline
        from .cloud_engine import CloudPipeline, CLOUD_MODEL
        pipe = CloudPipeline(language=args.language, on_update=on_update,
                             on_event=lambda e: broadcast({"event": e}))
        PIPE = pipe
        ENGINE.update(engine="cloud", model=CLOUD_MODEL,
                      language=args.language)
        print(f"CLOUD režim: {CLOUD_MODEL} (OpenAI Realtime)")
        if args.file:
            pipe.start_file(args.file)
        else:
            pipe.start_mic(os.environ.get("TITULKY_DEVICE") or None)
            print("Počúvam mikrofón — hovor.")
        try:
            asyncio.run(ws_main("localhost", args.port))
        except KeyboardInterrupt:
            pass
        finally:
            pipe.stop()
            print("Prepis:", " ".join(pipe.final_text))
            for name, xs in (("DRAFT", pipe.draft_latencies),
                             ("FINAL", pipe.final_latencies)):
                if xs:
                    xs2 = sorted(xs)
                    print(f"{name}: p50={xs2[len(xs2)//2]:.2f}s "
                          f"avg={sum(xs)/len(xs):.2f}s n={len(xs)}")
        return

    pipe = TwoTierPipeline(language=args.language, on_update=on_update,
                           on_event=lambda e: broadcast({"event": e}))
    PIPE = pipe
    ENGINE.update(engine="local", model=None, language=args.language)

    # run-trace: beh viditeľný v Projektovník BUILD tab Flow
    tr = rt.RunTrace("titulky", spustil="rucne")
    global AUDIO_DIR
    AUDIO_DIR = os.path.join(tr.dir, "audio")
    _t_ses = time.time()
    _tr_state: dict[str, int | None] = {"vstup_seq": None,
                                        "last_final_seq": None}

    def trace_event(e):
        # _audio a np polia nejdú do websocket JSON — odstráň pred broadcastom
        pub = {k: v for k, v in e.items() if not k.startswith("_")}
        zdroj = ""
        try:
            if e["t"] == "final":
                words = e["words"]
                seg = e.get("_audio")
                if seg is not None and len(seg):
                    import wave

                    import numpy as np
                    adir = os.path.join(tr.dir, "audio")
                    os.makedirs(adir, exist_ok=True)
                    zdroj = os.path.join(
                        adir, f"{e.get('audio_t0_s', 0):08.2f}s.wav")
                    with wave.open(zdroj, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(
                            (np.clip(seg, -1, 1) * 32767).astype("<i2")
                            .tobytes())
                    pub["wav"] = os.path.basename(zdroj)  # pre /audio/ v UI
                _tr_state["last_final_seq"] = tr.krok(
                    f"final: {words[:36]}" + ("…" if len(words) > 36 else ""),
                    t0=time.time(), stav="ok",
                    parent=_tr_state["vstup_seq"],
                    vystup=rt.vystup("text", words, zdroj=zdroj,
                                     pocet=len(words.split())),
                    meta={"lat_s": e.get("lat"),
                          "audio_t0_s": e.get("audio_t0_s"),
                          "audio_dur_s": e.get("audio_dur_s"),
                          "rms": e.get("rms")})
            elif e["t"] == "corr":
                tr.krok(f"korekcia: {e['from']} → {e['to']}",
                        t0=time.time(), stav="ok",
                        parent=_tr_state["last_final_seq"],
                        vystup=rt.vystup("diff",
                                         f"- {e['from']}\n+ {e['to']}"))
            elif e["t"] == "korekcia" and "zlyhalo" in e.get("msg", ""):
                tr.krok("azure volanie", t0=time.time(), stav="chyba",
                        parent=_tr_state["vstup_seq"],
                        chyba={"sprava": e["msg"], "stack": None})
        except OSError:
            pass  # trace nesmie zhodiť pipeline
        broadcast({"event": pub})

    pipe.on_event = trace_event
    print("Warmup modelov (base draft + kinit-turbo-sk final)…")
    _t0 = time.time()
    pipe.warmup()
    tr.krok("warmup modelov", t0=_t0,
            vystup=rt.vystup("modely", "base draft + kinit-turbo-sk final"))
    if args.file:
        pipe.play_file(args.file)
        _tr_state["vstup_seq"] = tr.krok(
            "vstup", t0=time.time(),
            vstupy=[rt.vstup("audio", typ="path", zdroj=args.file)])
    else:
        pipe.start_mic()
        _tr_state["vstup_seq"] = tr.krok(
            "vstup", t0=time.time(),
            vstupy=[rt.vstup("audio", typ="mic",
                             zdroj=os.environ.get("TITULKY_DEVICE",
                                                  "default"))])
        print("Počúvam mikrofón — hovor.")

    threading.Thread(target=pipe.run, daemon=True).start()
    try:
        asyncio.run(ws_main("localhost", args.port))
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        tr.krok("koniec behu", t0=_t_ses,
                vystup=rt.vystup("summary",
                                 " ".join(pipe.final_text)[:300],
                                 pocet=len(pipe.final_text),
                                 korekcie=pipe.corrections),
                meta={"flicker": round(pipe.final_agree.flicker_rate(), 3),
                      "final_lat_avg_s": round(statistics.mean(
                          pipe.final_latencies), 2)
                          if pipe.final_latencies else None,
                      "final_lat_max_s": round(max(pipe.final_latencies), 2)
                          if pipe.final_latencies else None,
                      "draft_lat_avg_s": round(statistics.mean(
                          pipe.draft_latencies), 2)
                          if pipe.draft_latencies else None})
        for name, xs in (("DRAFT", pipe.draft_latencies), ("FINAL", pipe.final_latencies)):
            if xs:
                print(f"\n{name} latencia avg={statistics.mean(xs):.2f}s "
                      f"p50={statistics.median(xs):.2f}s min={min(xs):.2f}s max={max(xs):.2f}s")
        print(f"Flicker rate: {pipe.final_agree.flicker_rate():.1%}")
        print("Prepis:", " ".join(pipe.final_text))


if __name__ == "__main__":
    main()
