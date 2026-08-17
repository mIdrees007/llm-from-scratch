import subprocess
import sys
import pathlib
import argparse
import shlex


ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"


def run(cmd: str):
    print(f"\n>>> {cmd}")

    res = subprocess.run(
        shlex.split(cmd),
        cwd=ROOT
    )

    if res.returncode != 0:
        sys.exit(res.returncode)


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--visualize",
        action="store_true",
        help="run visualization scripts and save PNGs to ./out"
    )

    args = p.parse_args()

    OUT.mkdir(exist_ok=True)

    # Sanity check
    run(f'"{sys.executable}" attn_numpy_demo.py')

    # Unit tests
    run(f'"{sys.executable}" -m pytest -q tests/test_attn_math.py')
    run(f'"{sys.executable}" -m pytest -q tests/test_causal_mask.py')

    # Matrix math walkthrough
    run(f'"{sys.executable}" demo_mha_shapes.py')

    # Optional visualization
    if args.visualize:
        run(f'"{sys.executable}" demo_visualize_multi_head.py')
        print(f"\nVisualization images saved to: {OUT}")

    print("\nAll Part 1 demos/tests completed.")


if __name__ == "__main__":
    main()