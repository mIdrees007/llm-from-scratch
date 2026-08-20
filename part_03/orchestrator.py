# cd part_03
# pyton orchestrator.py --demo
# python -q


import argparse, pathlib, subprocess, sys, shlex

ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>>> {cmd}")
    
    parts = shlex.split(cmd)

    if parts[0] == "python":
        parts[0] = sys.executable

    res = subprocess.run(parts, cwd=ROOT)

    if res.returncode != 0:
        sys.exit(res.returncode)
        
if __name__ == "__main__": 
    p = argparse.ArgumentParser()    
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    
    
    # 1 run unit tests 
    run('python -m pytest -q tests/test_rmsnorm.py')
    run('python -m pytest -q tests/test_rope_apply.py')
    run('python -m pytest -q tests/test_kvcache_shapes.py')
    
    # 2
    if args.demo:
        run("python demo_generate.py --rmsnorm --rope  --swiglu --sliding_window 64 --sink 4 --tokens 200")
        
    print("\n Part 3 checks completed")
        