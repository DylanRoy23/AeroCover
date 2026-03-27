import argparse
import os
import json
import pickle
from pathlib import Path
from datetime import datetime

DEFAULT_REPLAY_DIR = "replay"
DEFAULT_MAX_SIZE_MB = 1000


def get_replay_dir(base=DEFAULT_REPLAY_DIR):
    return Path(base)


def save_buffer(buffer_data, algo, task="coverage", tag="latest",
                base=DEFAULT_REPLAY_DIR, max_size_mb=DEFAULT_MAX_SIZE_MB):
    replay_dir = Path(base) / algo / task
    replay_dir.mkdir(parents=True, exist_ok=True)

    # Check total size before saving
    total_size = 0

    for f in Path(base).rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size    
    total_mb = total_size / (1024 * 1024)
    assert total_mb < max_size_mb, (
        f"Replay storage at {total_mb:.0f}MB exceeds limit of {max_size_mb}MB. "
        f"Run with --cleanup to free space."
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{tag}.pkl"
    filepath = replay_dir / filename

    with open(filepath, "wb") as f:
        pickle.dump(buffer_data, f)

    # Update metadata
    meta_path = replay_dir / "metadata.json"
    meta = []
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    meta.append({
        "file": filename,
        "timestamp": timestamp,
        "tag": tag,
        "size_mb": os.path.getsize(filepath) / (1024 * 1024),
        "algo": algo,
        "task": task,
    })

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {filepath} ({os.path.getsize(filepath)/1024:.0f} KB)")
    return filepath


def extract_sb3_buffer(model, n_samples=None):
    """Extract replay buffer contents from an SB3 off-policy model."""
    buf = model.replay_buffer
    if hasattr(buf, 'size'):
        size = buf.size()
    else:
        size = buf.buffer_size
        
    if n_samples and n_samples < size:
        indices = list(range(n_samples))
    else:
        indices = list(range(size))

    return {
        "observations": buf.observations[indices].copy(),
        "actions": buf.actions[indices].copy(),
        "rewards": buf.rewards[indices].copy(),
        "next_observations": buf.next_observations[indices].copy(),
        "dones": buf.dones[indices].copy(),
        "size": len(indices),
    }


def cleanup(base=DEFAULT_REPLAY_DIR, keep_newest=3):
    """Remove old replay files, keeping only the newest per algo/task."""
    base_path = Path(base)
    if not base_path.exists():
        print("No replay directory found.")
        return

    for algo_dir in sorted(base_path.iterdir()):
        if not algo_dir.is_dir():
            continue
        for task_dir in sorted(algo_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            pkl_files = sorted(task_dir.glob("*.pkl"), key=lambda f: f.stat().st_mtime)

            if len(pkl_files) <= keep_newest:
                continue

            to_remove = pkl_files[:-keep_newest]
            for f in to_remove:
                size = f.stat().st_size / 1024
                f.unlink()
                print(f"  Removed {f} ({size:.0f} KB)")


def status(base=DEFAULT_REPLAY_DIR):
    """Print summary of stored replay data."""
    base_path = Path(base)
    if not base_path.exists():
        print("No replay directory.")
        return

    total = 0
    for algo_dir in sorted(base_path.iterdir()):
        if not algo_dir.is_dir():
            continue
        for task_dir in sorted(algo_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            files = list(task_dir.glob("*.pkl"))
            size = sum(f.stat().st_size for f in files)
            total += size
            print(f"  {algo_dir.name}/{task_dir.name}: "
                  f"{len(files)} files, {size/1024/1024:.1f} MB")

    print(f"\nTotal: {total/1024/1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay buffer manager")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--keep-newest", type=int, default=3)
    parser.add_argument("--max-size-mb", type=int, default=DEFAULT_MAX_SIZE_MB)
    args = parser.parse_args()

    if args.status:
        status()
    elif args.cleanup:
        cleanup(keep_newest=args.keep_newest)
    else:
        status()