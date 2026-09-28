"""Vstupný bod: spustí WebSocket server + pipeline.

Použitie:
  python -m titulky.main                # živý mikrofón
  python -m titulky.main --file x.wav   # simulovaný live vstup zo súboru
  python -m titulky.main --model medium --language sk
Overlay: otvor http://localhost:8765 v prehliadači.
"""
import argparse
import asyncio
import json
import pathlib
import threading

import websockets
from websockets.http11 import Response
from websockets.datastructures import Headers

from .pipeline import LivePipeline

CLIENTS = set()
LOOP = None
HTML = (pathlib.Path(__file__).parent / "overlay.html").read_bytes()


def broadcast(data: dict):
    if LOOP is None:
        return
    msg = json.dumps(data, ensure_ascii=False)
    for ws in list(CLIENTS):
        asyncio.run_coroutine_threadsafe(ws.send(msg), LOOP)


async def handler(ws):
    CLIENTS.add(ws)
    try:
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)


def serve_html(connection, request):
    if request.path == "/":
        return Response(200, "OK", Headers({"Content-Type": "text/html; charset=utf-8"}), HTML)
    return None  # ostatné cesty = WebSocket upgrade


async def ws_main(host, port):
    global LOOP
    LOOP = asyncio.get_running_loop()
    async with websockets.serve(handler, host, port, process_request=serve_html):
        print(f"Overlay: http://{host}:{port}")
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="WAV 16 kHz mono namiesto mikrofónu")
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default="sk")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    def on_update(d):
        line = d["committed"][-80:]
        print(f"[{d['audio_s']:6.1f}s lat={d['latency_s'] and round(d['latency_s'],2)}] "
              f"{line} | {d['pending']}")
        broadcast(d)

    pipe = LivePipeline(model_size=args.model, language=args.language,
                        step_s=args.step, on_update=on_update)
    print("Načítavam model…")
    if args.file:
        pipe.play_file(args.file)
    else:
        pipe.start_mic()
        print("Počúvam mikrofón — hovor.")

    threading.Thread(target=pipe.run, daemon=True).start()
    try:
        asyncio.run(ws_main("localhost", args.port))
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        if pipe.latencies:
            import statistics
            print(f"\nLatencia avg={statistics.mean(pipe.latencies):.2f}s "
                  f"p50={statistics.median(pipe.latencies):.2f}s "
                  f"max={max(pipe.latencies):.2f}s")
        print(f"Flicker rate: {pipe.agree.flicker_rate():.1%}")
        print("Prepis:", " ".join(pipe.committed_text))


if __name__ == "__main__":
    main()
