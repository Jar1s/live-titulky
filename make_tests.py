"""Vygeneruje testovaciu sadu WAV + referencie (tests/).

Prípady: normal, rýchla reč, pomalá reč, tichá reč (gain 0.05),
šum (SNR ~10 dB), dlhé pauzy medzi vetami, čisté ticho (halucinačný test).
"""
import json
import os
import subprocess
import wave

import numpy as np

SR = 16000
os.makedirs("tests", exist_ok=True)

V1 = ("Automatické titulky prevádzajú živú reč na text v reálnom čase. "
      "Systém musí pracovať rýchlo a presne.")
V2 = ("Kvalita rozpoznávania závisí od hluku v okolí, výslovnosti "
      "a rýchlosti reči. Meranie presnosti je dôležité.")


def say(text, out, rate=None):
    cmd = ["say", "-v", "Laura", "-o", "/tmp/say.aiff"]
    if rate:
        cmd += ["-r", str(rate)]
    subprocess.run(cmd + [text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    "/tmp/say.aiff", out], check=True)


def load(p):
    wf = wave.open(p, "rb")
    return np.frombuffer(wf.readframes(wf.getnframes()),
                         dtype=np.int16).astype(np.float32) / 32768


def save(p, a):
    w = wave.open(p, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
    w.close()


refs = {}

# 1. normal
say(V1 + " " + V2, "tests/normal.wav")
refs["tests/normal.wav"] = V1 + " " + V2

# 2. rýchla reč
say(V1, "tests/fast.wav", rate=280)
refs["tests/fast.wav"] = V1

# 3. pomalá reč
say(V1, "tests/slow.wav", rate=120)
refs["tests/slow.wav"] = V1

# 4. tichá reč (gain 0.05)
save("tests/quiet.wav", load("tests/normal.wav") * 0.05)
refs["tests/quiet.wav"] = refs["tests/normal.wav"]

# 5. šum SNR ~10 dB
a = load("tests/normal.wav")
rng = np.random.default_rng(7)
noise = rng.normal(0, 1, len(a)).astype(np.float32)
snr = 10 ** (10 / 20)
noise *= float(np.std(a)) / (float(np.std(noise)) * snr)
save("tests/noisy.wav", a + noise)
refs["tests/noisy.wav"] = refs["tests/normal.wav"]

# 6. pauzy: veta — 3 s ticho — veta (blokový test + leak starých slov)
say(V1, "/tmp/p1.wav".replace("/tmp", "tests").replace("tests/p1", "tests/_p1"))
say(V2, "tests/_p2.wav")
gap = np.zeros(3 * SR, dtype=np.float32)
save("tests/pauses.wav", np.concatenate([load("tests/_p1.wav"), gap,
                                         load("tests/_p2.wav"), gap]))
refs["tests/pauses.wav"] = V1 + " " + V2
os.remove("tests/_p1.wav"); os.remove("tests/_p2.wav")

# 7. čisté ticho 15 s (očakávaný výstup: NIČ — halucinačný test)
save("tests/silence.wav", np.zeros(15 * SR, dtype=np.float32))
refs["tests/silence.wav"] = ""

json.dump(refs, open("tests/refs.json", "w"), ensure_ascii=False, indent=1)
print("OK:", list(refs))
