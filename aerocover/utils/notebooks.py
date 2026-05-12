from __future__ import annotations
from typing import Any, Dict

def print_kv(d: Dict[str, Any], float_fmt: str = "{:.3f}"):
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{k}: {float_fmt.format(v)}")
        else:
            print(f"{k}: {v}")