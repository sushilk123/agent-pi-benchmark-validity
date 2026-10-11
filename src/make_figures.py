"""Figures for the final paper, all from verified result JSONs."""
import matplotlib; matplotlib.use("Agg")
import sys  # noqa: E401
sys.path.insert(0, __import__('os').path.dirname(__file__))
from paths import ROOT, INJECAGENT_DIR, BIPIA_DIR, RESULTS_DIR, LOG_DIR  # noqa: E402,F401
import matplotlib.pyplot as plt
import json, os
H=str(RESULTS_DIR); OUT=str(ROOT / "figures"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.size":9,"figure.dpi":300,"axes.grid":True,"grid.alpha":0.3})
C={"claude":"#C03D3E","llama":"#3B6FB6","nova":"#3A923A","mistral":"#8161AC","deepseek":"#E1812C","other":"#7F7F7F"}
def col(m):
    ml=m.lower()
    for k in C:
        if k in ml: return C[k]
    return C["other"]

u=json.load(open(f"{H}/unified_results.json")); inj=u["injecagent"]; bip=u["bipia"]
SH={"Claude Opus 5":"Opus5","Claude Sonnet 5":"Son5","Claude Haiku 4.5":"Hk4.5","Claude 3 Haiku":"Hk3",
    "Llama 3.3 70B":"L70B","Llama 3.1 8B":"L8B","Nova Premier":"NvPrem","Nova Pro":"NvPro","Nova Micro":"NvMicro",
    "Mistral Pixtral L":"Mistral","DeepSeek R1":"DeepSeek"}
ms=[m for m in inj if inj[m]["asr"] is not None and m in bip and bip[m]["asr"] is not None]

# Fig A: scatter InjecAgent vs BIPIA
fig,ax=plt.subplots(figsize=(3.4,3.0))
xs=[inj[m]["asr"] for m in ms]; ys=[bip[m]["asr"] for m in ms]
for m in ms:
    ax.scatter(inj[m]["asr"],bip[m]["asr"],s=34,color=col(m),zorder=3)
    ax.annotate(SH.get(m,m),(inj[m]["asr"],bip[m]["asr"]),fontsize=5.6,xytext=(3,3),textcoords="offset points")
n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
b=sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/max(1e-9,sum((xs[i]-mx)**2 for i in range(n))); a=my-b*mx
xr=[min(xs),max(xs)]; ax.plot(xr,[a+b*x for x in xr],"--",color="#555",lw=1,zorder=2)
ax.set_xlabel("InjecAgent ASR (%) — tool-hijack"); ax.set_ylabel("BIPIA ASR (%) — content inject")
ax.set_title(r"Benchmarks disagree: $\tau$=$-$0.24 (n=11, p=0.39)",fontsize=7.5)
ax.set_xlim(-5,80); ax.set_ylim(-5,108)
plt.tight_layout(); fig.savefig(f"{OUT}/final_scatter.png",bbox_inches="tight"); plt.close(fig)

# Fig B: defense sweep grouped bars
dl=json.load(open(f"{H}/defense_local.json")); db=json.load(open(f"{H}/defense_bipia.json"))
rows=[]
for m,r in dl.items(): rows.append((m,r["none"]["asr"],r["instruction"]["asr"],r["spotlight"]["asr"]))
for m,r in db.items():
    if m.startswith("Claude"): continue
    rows.append((m,r["none"]["asr"],r["instruction"]["asr"],r["spotlight"]["asr"]))
rows.sort(key=lambda z:-z[1])
labels=[r[0].replace(" Pixtral L","").replace("Qwen2.5","Qwen") for r in rows]
none=[r[1] for r in rows]; ins=[r[2] for r in rows]; spo=[r[3] for r in rows]
import numpy as np
xidx=np.arange(len(rows)); w=0.26
fig,ax=plt.subplots(figsize=(7.0,2.8))
ax.bar(xidx-w,none,w,label="No defense",color="#C03D3E")
ax.bar(xidx,ins,w,label="Instruction",color="#3B6FB6")
ax.bar(xidx+w,spo,w,label="Spotlight",color="#3A923A")
ax.set_ylabel("BIPIA ASR (%)"); ax.set_xticks(xidx); ax.set_xticklabels(labels,rotation=35,ha="right",fontsize=6.5)
ax.legend(fontsize=7,frameon=False,ncol=3); ax.set_title("Defenses do not generalize across models",fontsize=8.5)
plt.tight_layout(); fig.savefig(f"{OUT}/final_defense.png",bbox_inches="tight"); plt.close(fig)

# Fig C: AgentDojo utility vs ASR
ad=json.load(open(f"{H}/agentdojo_results.json"))
am=[m for m in ad if "error" not in ad[m]]
util=[ad[m]["utility_pct"] for m in am]; asr=[ad[m]["asr_pct"] for m in am]
xidx=np.arange(len(am)); w=0.38
fig,ax=plt.subplots(figsize=(3.4,2.6))
ax.bar(xidx-w/2,util,w,label="Utility",color="#3A923A")
ax.bar(xidx+w/2,asr,w,label="ASR",color="#C03D3E")
ax.set_ylabel("%"); ax.set_xticks(xidx); ax.set_xticklabels([SH.get(m,m) for m in am],fontsize=6.5)
ax.legend(fontsize=7,frameon=False); ax.set_title("AgentDojo: utility vs. attack success",fontsize=8)
ax.set_ylim(0,108)
plt.tight_layout(); fig.savefig(f"{OUT}/final_agentdojo.png",bbox_inches="tight"); plt.close(fig)
print("wrote final_scatter.png, final_defense.png, final_agentdojo.png")
