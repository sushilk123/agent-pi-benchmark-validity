"""
$0 local-model extension of the unified runner, via Ollama. Runs additional
(free, local) models on both benchmarks and APPENDS to the SAME checkpoint
(unified_ckpt.jsonl) so unified_runner.aggregate() includes them — boosting the
model count toward statistical significance without any API spend.

Prereq: `ollama serve` running; models pulled. NOT FOR DEPLOYMENT.
Usage: python3 unified_local.py --n 40 [--run]  then  unified_runner.py --aggregate
"""
import argparse, json, os, sys, urllib.request
import sys  # noqa: E401
sys.path.insert(0, __import__('os').path.dirname(__file__))
from paths import ROOT, INJECAGENT_DIR, BIPIA_DIR, RESULTS_DIR, LOG_DIR  # noqa: E402,F401
sys.path.insert(0, str(INJECAGENT_DIR))
from agentsec_pilot import get_tool_dict, build_prompt as inj_prompt, PROMPT_DICT
from src.output_parsing import evaluate_output_prompted
from agentsec_bipia_saturation import load_contexts, build_prompt as bip_prompt, INJECT_REF, SYS as BIP_SYS

INJ_DATA = str(INJECAGENT_DIR / "data")
CKPT = str(RESULTS_DIR / "unified_ckpt.jsonl")
INJ_SYS, INJ_USER = PROMPT_DICT["InjecAgent"]

# Local models (Ollama tags) -> display labels. All free, run on-device.
LOCAL = {
    "Llama 3.1 8B (local)": "llama3.1:8b",
    "Mistral 7B (local)": "mistral:7b",
    "Qwen2.5 7B (local)": "qwen2.5:7b",
    "Gemma 2 9B (local)": "gemma2:9b",
    "Qwen2.5 14B (local)": "qwen2.5:14b",
    "Phi-4 14B (local)": "phi4:latest",
}


def ollama_chat(model, system, user, max_tokens):
    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"num_predict": max_tokens}}
    try:
        req = urllib.request.Request("http://localhost:11434/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=180).read())
        return r["message"]["content"]
    except Exception as e:
        print(f"    ollama err {model}: {str(e)[:70]}", file=sys.stderr)
        return None


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
    with open(CKPT, "a") as ck:
        for run_id in range(R):
            for label, tag in LOCAL.items():
                for idx, item in enumerate(inj_cases):
                    if ("injecagent", label, run_id, idx) in done:
                        continue
                    out = ollama_chat(tag, INJ_SYS, inj_prompt(INJ_SYS, INJ_USER, item, tool_dict), 1024)
                    res = "error" if out is None else evaluate_output_prompted(
                        out, item["Attacker Tools"][0], item["Attacker Tools"],
                        item["User Tool"], item["Attacker Instruction"])["eval"]
                    ck.write(json.dumps({"bench": "injecagent", "model": label, "run": run_id,
                                         "idx": idx, "result": res}) + "\n"); ck.flush()
                for idx, (ctx, q) in enumerate(bip_cases):
                    if ("bipia", label, run_id, idx) in done:
                        continue
                    out = ollama_chat(tag, BIP_SYS, bip_prompt(ctx, q), 512)
                    res = ("error" if out is None else
                           ("succ" if INJECT_REF.lower() in out.lower() else "unsucc"))
                    ck.write(json.dumps({"bench": "bipia", "model": label, "run": run_id,
                                         "idx": idx, "result": res}) + "\n"); ck.flush()
                print(f"  run{run_id} {label} done", file=sys.stderr)
    print("Local models appended to checkpoint. Run: python3 unified_runner.py --aggregate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        run(a.n, a.runs)
    else:
        print(f"Dry: {len(LOCAL)} local models x {a.n} cases x {a.runs} runs x 2 benchmarks "
              f"= {len(LOCAL)*a.n*a.runs*2} local calls (free, ~7s each). Add --run.")
