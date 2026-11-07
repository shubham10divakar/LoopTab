"""Download the benchmark datasets with the Kaggle API (design doc Section 5.1).

Requires ``pip install kaggle`` and an API token in ~/.kaggle/kaggle.json.
IEEE-CIS is a competition dataset: you must accept its rules on the Kaggle
website before the download works.

    python -m looptab.data.download --dataset ccfraud --out datasets/ccfraud
    python -m looptab.data.download --dataset all --out datasets
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

SOURCES = {
    "ccfraud": ("dataset", "mlg-ulb/creditcardfraud"),
    "paysim": ("dataset", "ealaxi/paysim1"),
    "fraud_ecom": ("dataset", "vbinh002/fraud-ecommerce"),
    "ieee_cis": ("competition", "ieee-fraud-detection"),
}


def download(name: str, out_dir: str | Path) -> Path:
    kind, ref = SOURCES[name]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if kind == "dataset":
        cmd = ["kaggle", "datasets", "download", "-d", ref, "-p", str(out), "--unzip"]
    else:
        cmd = ["kaggle", "competitions", "download", "-c", ref, "-p", str(out)]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    for z in out.glob("*.zip"):  # competitions do not support --unzip
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
        z.unlink()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=[*SOURCES, "all"])
    p.add_argument("--out", default="datasets", help="output dir (per-dataset subdir when 'all')")
    args = p.parse_args(argv)
    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    failed = []
    for n in names:
        target = Path(args.out) / n if args.dataset == "all" else Path(args.out)
        try:
            download(n, target)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            failed.append(n)
            print(f"[error] {n}: {e}")
            if SOURCES[n][0] == "competition":
                print(f"  A 403 here usually means the competition rules are not accepted yet.\n"
                      f"  Open https://www.kaggle.com/competitions/{SOURCES[n][1]}/rules, click\n"
                      f"  'I Understand and Accept', then rerun with --dataset {n}.")
    if failed:
        print(f"failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
