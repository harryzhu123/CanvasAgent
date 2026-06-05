"""
Auto-detect and occupy free GPUs incrementally.
Scans every few seconds, grabs free GPUs one-by-one via subprocesses until target count is met.

Usage: python occupy_gpus.py [--num-gpus 4] [--target-pct 0.85] [--free-threshold 0.80] [--interval 10]
"""

import subprocess
import time
import signal
import sys
import os
import argparse
import multiprocessing


def get_gpu_info():
    """Query nvidia-smi for all GPUs, return list of (index, total_mb, free_mb)."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        idx, total_mb, free_mb = int(parts[0]), float(parts[1]), float(parts[2])
        gpus.append((idx, total_mb, free_mb))
    return gpus


def occupy_single_gpu(gpu_idx, target_pct, ready_event):
    """Worker function: occupy one GPU in a separate process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    import torch
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    target = int(total * target_pct)
    num_elements = target // 4
    t = torch.empty(num_elements, dtype=torch.float32, device="cuda:0")
    actual_mb = t.nelement() * t.element_size() / (1024 ** 2)
    total_mb = total / (1024 ** 2)
    print(f"  [pid={os.getpid()}] GPU {gpu_idx} ({props.name}): "
          f"{actual_mb:.0f}/{total_mb:.0f} MB ({actual_mb/total_mb*100:.1f}%) - OCCUPIED")
    ready_event.set()
    # Hold forever until parent kills us
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)


def main():
    parser = argparse.ArgumentParser(description="Auto-detect and occupy free GPUs incrementally")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to occupy")
    parser.add_argument("--target-pct", type=float, default=0.75, help="VRAM percentage to occupy (0-1)")
    parser.add_argument("--free-threshold", type=float, default=0.75,
                        help="Minimum free VRAM ratio to consider a GPU as 'free' (0-1)")
    parser.add_argument("--interval", type=int, default=5, help="Scan interval in seconds")
    args = parser.parse_args()

    print(f"Config: need {args.num_gpus} GPUs, occupy {args.target_pct*100:.0f}% VRAM, "
          f"free threshold >{args.free_threshold*100:.0f}%, scan every {args.interval}s")
    print("=" * 60)

    occupied = {}  # gpu_idx -> Process
    scan_count = 0

    def cleanup(*_):
        print("\nCleaning up...")
        for idx, proc in occupied.items():
            proc.terminate()
        for idx, proc in occupied.items():
            proc.join(timeout=3)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    while len(occupied) < args.num_gpus:
        scan_count += 1
        gpu_info = get_gpu_info()
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Scan #{scan_count} - occupied {len(occupied)}/{args.num_gpus} {list(occupied.keys())}")

        for idx, total_mb, free_mb in gpu_info:
            pct = free_mb / total_mb * 100
            status = "HELD" if idx in occupied else ("FREE" if pct >= args.free_threshold * 100 else "busy")
            print(f"  GPU {idx}: {free_mb:.0f}/{total_mb:.0f} MB free ({pct:.1f}%) [{status}]")

        # Find new free GPUs not already occupied
        new_targets = []
        for idx, total_mb, free_mb in gpu_info:
            if idx in occupied:
                continue
            if free_mb / total_mb >= args.free_threshold:
                new_targets.append(idx)

        # Sort by free memory descending, take only what we still need
        remaining = args.num_gpus - len(occupied)
        new_targets = sorted(new_targets, key=lambda i: -dict((g[0], g[2]) for g in gpu_info)[i])
        new_targets = new_targets[:remaining]

        for gpu_idx in new_targets:
            print(f"  -> Grabbing GPU {gpu_idx}...")
            ready = multiprocessing.Event()
            p = multiprocessing.Process(target=occupy_single_gpu,
                                        args=(gpu_idx, args.target_pct, ready),
                                        daemon=True)
            p.start()
            ready.wait(timeout=30)
            if p.is_alive():
                occupied[gpu_idx] = p
            else:
                print(f"  -> GPU {gpu_idx}: worker died, skipping")

        if len(occupied) < args.num_gpus:
            still_need = args.num_gpus - len(occupied)
            print(f"  -> Still need {still_need} more, retrying in {args.interval}s...\n")
            time.sleep(args.interval)

    print(f"\n{'='*60}")
    print(f"SUCCESS: Occupied {args.num_gpus} GPUs: {sorted(occupied.keys())}")
    print(f"Each at {args.target_pct*100:.0f}% VRAM. Press Ctrl+C to release and exit.")
    print(f"{'='*60}")

    # Wait for Ctrl+C
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
