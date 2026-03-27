import json
import torch
from pathlib import Path
from dataclasses import asdict


def save_checkpoint(model, config, algo, task="coverage",
                    base="checkpoints", tag="best"):
    ckpt_dir = Path(base) / algo / task / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    if hasattr(model, "save"):
        model.save(str(ckpt_dir / "model"))
    else:
        torch.save(model.state_dict(), ckpt_dir / "model.pt")

    # Save config
    if hasattr(config, '__dataclass_fields__'):
        cfg_dict = asdict(config)
    else:
        cfg_dict = vars(config)    
        
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    print(f"  Saved {algo}/{task}/{tag}")