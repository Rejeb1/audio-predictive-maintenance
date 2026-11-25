"""Run every experiment; finished runs are skipped, so it can be restarted at any time."""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from asd.config import MACHINES, RESULTS

DATASETS = ("dcase", "mimii_-6dB", "mimii_0dB", "mimii_6dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--epochs", type=int, help="override, for quick tests only")
    a = ap.parse_args()
    failed = []
    for ds in a.datasets:
        for m in MACHINES:
            for kind in ("panns", "dense", "conv"):
                name = f"{ds}_{m}_{kind}"
                if (RESULTS / "scores" / f"{name}.csv").exists():
                    print("skip (done):", name, flush=True)
                    continue
                if kind == "panns":
                    cmd = ["run_embeddings.py"]
                else:
                    cmd = ["train_ae.py", "--kind", kind]
                    cmd += ["--epochs", str(a.epochs)] if a.epochs else []
                print(f"\n=== {name}", flush=True)
                cmd = [sys.executable, "-u", str(HERE / cmd[0]), *cmd[1:],
                       "--dataset", ds, "--machine", m]
                if subprocess.run(cmd).returncode:
                    failed.append(name)
    print("\nFAILED: " + ", ".join(failed) if failed else "\nAll runs complete.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
