"""Generate split JSONs for remaining (not yet completed) samples.

Usage:
    python tools/split_remaining.py --sub 02 --num_parts 2

Outputs:
    sub02_remaining_part0.json, sub02_remaining_part1.json, ...
"""
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 02")
    parser.add_argument("--num_parts", type=int, default=2)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.dataset_root is None:
        from local_config import get_paths
        args.dataset_root = get_paths()["dataset_root"]

    json_path = os.path.join(args.dataset_root, f"sub-00{args.sub}_test_va.json")
    result_dir = f"results/brain_va_5b_sub{args.sub}"

    with open(json_path) as f:
        data = json.load(f)

    done = set(os.listdir(result_dir)) if os.path.isdir(result_dir) else set()
    remaining = [d for d in data if os.path.basename(d["video"]).split(".")[0] + ".mp4" not in done]

    print(f"sub-{args.sub}: total={len(data)}, done={len(done)}, remaining={len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return

    out_dir = args.output_dir or "."
    chunk_size = (len(remaining) + args.num_parts - 1) // args.num_parts
    for i in range(args.num_parts):
        chunk = remaining[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            continue
        out_path = os.path.join(out_dir, f"sub{args.sub}_remaining_part{i}.json")
        with open(out_path, "w") as f:
            json.dump(chunk, f)
        print(f"  part{i}: {len(chunk)} samples -> {out_path}")

if __name__ == "__main__":
    main()
