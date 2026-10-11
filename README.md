# Are LLM-Agent Prompt-Injection Benchmarks Still Valid?

Code and data for **"Are LLM-Agent Prompt-Injection Benchmarks Still Valid? A Cross-Model and
Cross-Benchmark Study"** (Sushil Khairnar, Virginia Tech).

The harness measures indirect prompt-injection attack success rate (ASR) for many LLMs under a
single, error-audited decoding configuration across three benchmarks (InjecAgent, a targeted
BIPIA-email injection, and AgentDojo), plus a cross-model sweep of two prompt-level defenses.

## Repository layout

| Path | Contents |
|---|---|
| `src/unified_runner.py` | Main cross-model run: InjecAgent + BIPIA over 11 models (Amazon Bedrock `converse`), checkpointed and resumable; `--aggregate` rebuilds the results table |
| `src/unified_local.py` | Same benchmarks on free local models via Ollama |
| `src/agentsec_defense_bipia.py`, `src/agentsec_defense_local.py` | Defense sweep (none / instruction-defense / spotlighting), Bedrock and local |
| `src/agentdojo_bedrock_backend.py`, `src/run_agentdojo.py` | AgentDojo backend routed through Bedrock, and the scoped AgentDojo run |
| `src/agentsec_pilot.py`, `src/agentsec_bipia_saturation.py` | Benchmark prompt assembly and scoring helpers (reuse each benchmark's own success criterion) |
| `src/make_figures.py` | Regenerates the paper figures from `results/` |
| `results/unified_ckpt.jsonl` | Raw per-case outcomes (benchmark, model, run, case, result) |
| `results/*.json` | Aggregated results reported in the paper |
| `figures/` | Paper figures |

## Setup

```bash
pip install -r requirements.txt
./scripts/fetch_benchmarks.sh          # clones InjecAgent and BIPIA into ./benchmarks
```

Bedrock runs use the standard AWS credential chain (e.g. `aws configure` or an AWS profile) and
need model access enabled for the evaluated models in your account; set `AWS_REGION` if not
`us-west-2`. Local runs need [Ollama](https://ollama.com) with the models listed in
`src/unified_local.py` pulled.

## Reproducing

```bash
python3 src/unified_runner.py --aggregate     # rebuild Table 1 from the committed raw results (no API calls)
python3 src/make_figures.py                   # regenerate figures

python3 src/unified_runner.py --n 50 --runs 2 # dry run: prints the call count
python3 src/unified_runner.py --n 50 --runs 2 --run   # full re-run (incurs Bedrock cost)
python3 src/agentsec_defense_local.py --n 20 --run    # defense sweep on local models (free)
python3 src/run_agentdojo.py --run                    # scoped AgentDojo run
```

Every script is a dry run unless `--run` is passed. Failed API calls are recorded as errors and
excluded from ASR, never scored as a non-attack.

## Notes on the data

- All Bedrock results use provider-default decoding (no explicit temperature), the only setting
  valid for every evaluated vendor.
- `results/defense_bipia.json` contains Claude rows from a run in which those calls failed; they
  are **excluded** from the paper's defense analysis. The non-Claude rows are valid.
- BIPIA results use one deterministic targeted attack (exact-sentence match) on the first 50
  email contexts, not BIPIA's full attack suite.
- AgentDojo uses a text-prompted backend that yields low task utility for non-Claude models; treat
  its cross-model ASR as supporting evidence only (see the paper's limitations).

## Third-party benchmarks

This repository does **not** redistribute the benchmarks. Obtain them from their sources under
their own licenses:
InjecAgent (Zhan et al., 2024, https://github.com/uiuc-kang-lab/InjecAgent),
BIPIA (Yi et al., 2023, https://github.com/microsoft/BIPIA), and
AgentDojo (Debenedetti et al., 2024, `pip install agentdojo`).

## License

MIT (see `LICENSE`). Applies to the code and result files in this repository only.
