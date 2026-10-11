"""Repo-relative paths. Benchmarks are fetched into ./benchmarks by scripts/fetch_benchmarks.sh."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
INJECAGENT_DIR = BENCH_DIR / "InjecAgent"
BIPIA_DIR = BENCH_DIR / "BIPIA"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
