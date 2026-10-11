"""
Unified, config-controlled runner for the prompt-injection benchmark-validity study.

Design principles (fixing the issues found during exploration):
- ONE decoding path for all vendors: Bedrock `converse`, provider-DEFAULT decoding
  (no explicit temperature — Claude 5 rejects temperature via converse, so this is
  the only setting valid for every model → like-for-like comparison).
- Per-call error tracking: a failed call is recorded as "error" and EXCLUDED from
  ASR, never silently scored as 0.
- Repeated runs (R) per cell to quantify run-to-run (sampling) variance, since
  deterministic decoding is unavailable for Claude 5.
- Checkpoint JSONL: every (bench, model, run, idx) result is written and skipped on
  resume, so the multi-hour matrix survives credential refreshes.

NOT FOR DEPLOYMENT — local research script; creds via boto3.
Aggregate with: python3 unified_runner.py --aggregate
"""
import argparse, json, os, sys
import sys  # noqa: E401
sys.path.insert(0, __import__('os').path.dirname(__file__))
from paths import ROOT, INJECAGENT_DIR, BIPIA_DIR, RESULTS_DIR, LOG_DIR  # noqa: E402,F401
sys.path.insert(0, str(INJECAGENT_DIR))
import boto3
from agentsec_pilot import get_tool_dict, build_prompt as inj_prompt, PROMPT_DICT
from src.output_parsing import evaluate_output_prompted
from agentsec_bipia_saturation import load_contexts, build_prompt as bip_prompt, INJECT_REF, SYS as BIP_SYS
from common import bootstrap_ci

INJ_DATA = str(INJECAGENT_DIR / "data")
CKPT = str(RESULTS_DIR / "unified_ckpt.jsonl")
OUT = str(RESULTS_DIR / "unified_results.json")

MODELS = {
    "Claude Opus 5": "us.anthropic.claude-opus-5",
    "Claude Sonnet 5": "us.anthropic.claude-sonnet-5",
    "Claude Haiku 4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Llama 3.3 70B": "us.meta.llama3-3-70b-instruct-v1:0",
    "Llama 3.1 8B": "us.meta.llama3-1-8b-instruct-v1:0",
    "Nova Premier": "us.amazon.nova-premier-v1:0",
    "Nova Pro": "us.amazon.nova-pro-v1:0",
    "Nova Micro": "us.amazon.nova-micro-v1:0",
    "Mistral Pixtral L": "us.mistral.pixtral-large-2502-v1:0",
    "DeepSeek R1": "us.deepseek.r1-v1:0",
}
INJ_SYS, INJ_USER = PROMPT_DICT["InjecAgent"]


def converse(brt, mid, system, user, max_tokens):
    try:
        r = brt.converse(modelId=mid, system=[{"text": system}],
                         messages=[{"role": "user", "content": [{"text": user}]}],
                         inferenceConfig={"maxTokens": max_tokens})
        return " ".join(p.get("text", "") for p in r["output"]["message"]["content"])
    except Exception as e:
        print(f"    err {mid}: {str(e)[:70]}", file=sys.stderr)
        return None


def score_injecagent(out, item):
    if out is None:
        return "error"
    return evaluate_output_prompted(out, item["Attacker Tools"][0], item["Attacker Tools"],
                                    item["User Tool"], item["Attacker Instruction"])["eval"]


def score_bipia(out):
    if out is None:
        return "error"
    return "succ" if INJECT_REF.lower() in out.lower() else "unsucc"


def run(n, R):
    tool_dict = get_tool_dict()
    inj_cases = json.load(open(f"{INJ_DATA}/test_cases_dh_base.json"))[:n]
    bip_cases = load_contexts(n)
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT):
            try:
                r = json.loads(line); done.add((r["bench"], r["model"], r["run"], r["idx"]))
            except Exception:
                pass
    brt = boto3.client("bedrock-runtime", region_name="us-west-2")
    with open(CKPT, "a") as ck:
        for run_id in range(R):
            for mlabel, mid in MODELS.items():
                for idx, item in enumerate(inj_cases):
                    key = ("injecagent", mlabel, run_id, idx)
                    if key in done:
                        continue
                    out = converse(brt, mid, INJ_SYS, inj_prompt(INJ_SYS, INJ_USER, item, tool_dict), 1024)
                    ck.write(json.dumps({"bench": "injecagent", "model": mlabel, "run": run_id,
                                         "idx": idx, "result": score_injecagent(out, item)}) + "\n")
                    ck.flush()
                for idx, (ctx, q) in enumerate(bip_cases):
                    key = ("bipia", mlabel, run_id, idx)
                    if key in done:
                        continue
                    out = converse(brt, mid, BIP_SYS, bip_prompt(ctx, q), 512)
                    ck.write(json.dumps({"bench": "bipia", "model": mlabel, "run": run_id,
                                         "idx": idx, "result": score_bipia(out)}) + "\n")
                    ck.flush()
                print(f"  run{run_id} {mlabel} done", file=sys.stderr)
    aggregate()


def aggregate():
    rows = [json.loads(l) for l in open(CKPT) if l.strip()]
    out = {"config": {"decoding": "converse provider-default (no temperature)",
                      "note": "errors excluded from ASR; ASR over validly-scored cases"}}
    for bench in ["injecagent", "bipia"]:
        out[bench] = {}
        for mlabel in MODELS:
            cell = [r for r in rows if r["bench"] == bench and r["model"] == mlabel]
            if not cell:
                continue
            succ = sum(r["result"] == "succ" for r in cell)
            unsucc = sum(r["result"] == "unsucc" for r in cell)
            invalid = sum(r["result"] == "invalid" for r in cell)
            error = sum(r["result"] == "error" for r in cell)
            valid = [1 if r["result"] == "succ" else 0 for r in cell if r["result"] in ("succ", "unsucc")]
            ci = bootstrap_ci(valid) if valid else [None, None, None]
            # per-run ASR for cross-run std
            runs = sorted({r["run"] for r in cell})
            per_run = []
            for rn in runs:
                rc = [r for r in cell if r["run"] == rn and r["result"] in ("succ", "unsucc")]
                if rc:
                    per_run.append(100 * sum(r["result"] == "succ" for r in rc) / len(rc))
            spread = [round(min(per_run), 1), round(max(per_run), 1)] if per_run else [None, None]
            out[bench][mlabel] = {"asr": ci[0], "ci95": [ci[1], ci[2]], "per_run_asr": [round(x, 1) for x in per_run],
                                  "run_spread": spread, "succ": succ, "unsucc": unsucc,
                                  "invalid": invalid, "error": error,
                                  "validity_pct": round(100 * (succ + unsucc) / max(1, succ + unsucc + invalid), 1)}
    json.dump(out, open(OUT, "w"), indent=2)
    # print
    print("\n=== UNIFIED (converse, provider-default; errors excluded) ===")
    print(f"{'Model':18}{'InjecAgent ASR':>16}{'valid%':>8}{'BIPIA ASR':>12}{'BIPIA runs':>14}")
    for m in MODELS:
        i = out["injecagent"].get(m, {}); b = out["bipia"].get(m, {})
        ia = f"{i.get('asr','-')}" if i.get('asr') is not None else "-"
        ba = f"{b.get('asr','-')}" if b.get('asr') is not None else "-"
        rs = f"{b.get('run_spread',['-','-'])}"
        print(f"{m:18}{ia:>16}{i.get('validity_pct','-'):>8}{ba:>12}{rs:>14}")
    print("Saved:", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    elif a.run:
        run(a.n, a.runs)
    else:
        total = a.n * len(MODELS) * a.runs * 2
        print(f"Dry run: {a.n} cases x {len(MODELS)} models x {a.runs} runs x 2 benchmarks "
              f"= {total} calls. Add --run.")
