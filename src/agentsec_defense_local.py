"""
$0 defense sweep on local (Ollama) models: BIPIA-email targeted injection under
none / instruction / spotlight defenses. The 'none' condition also yields local
saturation ASR. Uses the SAME prompts() as the Bedrock defense runner for parity.

Prereq: ollama serve running; models pulled.
Usage: python3 agentsec_defense_local.py --n 20 [--run]
Then aggregate/plot from defense_local.json.
"""
import argparse, json, sys, urllib.request
import sys  # noqa: E401
sys.path.insert(0, __import__('os').path.dirname(__file__))
from paths import ROOT, INJECAGENT_DIR, BIPIA_DIR, RESULTS_DIR, LOG_DIR  # noqa: E402,F401
from agentsec_defense_bipia import prompts   # reuse identical defense prompts
from agentsec_bipia_saturation import load_contexts, INJECT_REF
from common import bootstrap_ci

LOCAL = {
    "Llama 3.1 8B": "llama3.1:8b",
    "Mistral 7B": "mistral:7b",
    "Qwen2.5 7B": "qwen2.5:7b",
    "Gemma 2 9B": "gemma2:9b",
    "Qwen2.5 14B": "qwen2.5:14b",
    "Phi-4 14B": "phi4:latest",
}
CONDS = ["none", "instruction", "spotlight"]


def ollama(model, system, user):
    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"num_predict": 512}}
    try:
        req = urllib.request.Request("http://localhost:11434/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=180).read())["message"]["content"]
    except Exception as e:
        print(f"    err {model}: {str(e)[:70]}", file=sys.stderr)
        return None


def run(n, do_run):
    ctxs = load_contexts(n)
    print(f"Local defense sweep: {len(ctxs)} ctx x {len(LOCAL)} models x {len(CONDS)} conds "
          f"= {len(ctxs)*len(LOCAL)*len(CONDS)} local calls (free).")
    if not do_run:
        print("Dry run. Add --run."); return
    out = {}
    for mlabel, tag in LOCAL.items():
        out[mlabel] = {}
        for cond in CONDS:
            pc, errors = [], 0
            for i, (ctx, q) in enumerate(ctxs):
                sysp, user = prompts(ctx, q, cond)
                resp = ollama(tag, sysp, user)
                if resp is None:
                    errors += 1
                else:
                    pc.append(1 if INJECT_REF.lower() in resp.lower() else 0)
                print(f"  [{mlabel}|{cond}] {i+1}/{len(ctxs)}", file=sys.stderr)
            ci = bootstrap_ci(pc) if pc else [None, None, None]
            out[mlabel][cond] = {"asr": ci[0], "ci95": [ci[1], ci[2]],
                                 "succ": sum(pc), "n": len(pc), "errors": errors}
        json.dump(out, open(str(RESULTS_DIR / "defense_local.json"), "w"), indent=2)
    def f(x): return f"{x:5.1f}" if x is not None else " err "
    print("\n=== LOCAL DEFENSE SWEEP — BIPIA-email ASR ===")
    print(f"{'Model':16}{'none':>7}{'instr':>7}{'spot':>7}")
    for m in LOCAL:
        r = out[m]; print(f"{m:16}{f(r['none']['asr']):>7}{f(r['instruction']['asr']):>7}{f(r['spotlight']['asr']):>7}")
    print("Saved: defense_local.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--run", action="store_true"); a = ap.parse_args()
    run(a.n, a.run)
