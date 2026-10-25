"""Kvantizuj MLX whisper model (4-bit) bez torch — zrkadlí
mlx-examples/whisper/convert.py quantize() + load_models.load_model().

Použitie: .venv/bin/python tools/quantize_mlx.py SRC_DIR DST_DIR [BITS]
"""
import json
import pathlib
import shutil
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_whisper import whisper


def main():
    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    bits = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    group = 64

    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization", None)
    dims = whisper.ModelDimensions(
        **{k: v for k, v in cfg.items()
           if k in whisper.ModelDimensions.__dataclass_fields__})
    model = whisper.Whisper(dims, dtype=mx.float16)
    weights = mx.load(str(src / "weights.safetensors"))
    model.load_weights(list(weights.items()))

    nn.quantize(model, group, bits,
                class_predicate=lambda p, m: isinstance(m, nn.Linear)
                and hasattr(m, "to_quantized"))
    cfg["quantization"] = {"group_size": group, "bits": bits}

    dst.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst / "weights.safetensors"),
                        dict(tree_flatten(model.parameters())),
                        metadata={"format": "mlx"})
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    # mlx_whisper hľadá weights.safetensors; model.safetensors netreba
    print("hotovo:", dst, f"{bits}bit",
          f"{sum(f.stat().st_size for f in dst.iterdir())/1e6:.0f} MB")


if __name__ == "__main__":
    main()
