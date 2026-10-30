#!/bin/bash
#SBATCH --job-name=trainer_v2_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --account=maom
#SBATCH --partition=maom-h200
#SBATCH --output=trainer_v2_smoke_%j.log
#
# GPU smoke test for trainer v2 (docs/MIGRATION_TRAINER_V2.md) on a small D4
# subset of the LSD_finetuning manifests. Trains, for 2 epochs each:
#   lora_v2      default flags (affinity_heads_pairformer, ensemble output)
#   lora_legacy  --legacy-trainer --target-spec heads_pairformer
#   ft_v2        boltz finetune, affinity_heads_pairformer, with --val-csv
# then checks, from the artifacts themselves, which scopes actually changed,
# records GPU memory / wall time per arm, and rescores a few validation poses
# with vanilla vs lora_v2 (SUCCESS counts must match).
#
# Everything is written under RUN_DIR; LSD adapters/finetunes/caches are not
# touched. Submit from anywhere:  sbatch scripts/verify/trainer_v2_smoke.slurm
# Override: N_TRAIN, N_VAL, N_RESCORE, LSD_ROOT, RUN_DIR, CONDA_ENV.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." 2>/dev/null && pwd || echo /home/limcaoco/turbo/limcaoco/Boltz_affinity)"
[[ -f "${REPO}/src/boltz/lora/train.py" ]] || REPO=/home/limcaoco/turbo/limcaoco/Boltz_affinity
LSD_ROOT="${LSD_ROOT:-/home/limcaoco/turbo/limcaoco/LSD_finetuning}"
RUN_DIR="${RUN_DIR:-/scratch/maom_root/maom/limcaoco/trainer_v2_smoke_${SLURM_JOB_ID:-local}}"
N_TRAIN="${N_TRAIN:-64}"
N_VAL="${N_VAL:-32}"
N_RESCORE="${N_RESCORE:-20}"
CONDA_ENV="${CONDA_ENV:-boltz_affinity}"

mkdir -p "${RUN_DIR}"/{lora,finetune,cache,rescore_in,scores}
cd "${RUN_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export BOLTZ_CACHE=/home/limcaoco/turbo/limcaoco/.boltz
export BOLTZ_MSA_CACHE_DIR="${LSD_ROOT}/msa"
export BOLTZ_LORA_DIR="${RUN_DIR}/lora"
export BOLTZ_FINETUNE_DIR="${RUN_DIR}/finetune"
export BOLTZ_TRAIN_CACHE_DIR="${RUN_DIR}/cache"

echo "repo:   ${REPO} @ $(git -C "${REPO}" rev-parse --short HEAD) (dirty: $(git -C "${REPO}" status --porcelain --untracked-files=no | wc -l) files)"
echo "run:    ${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# ── Subset manifests (keep whole assay groups at the head of the file) ──────
python - "${LSD_ROOT}/manifests/D4" "${N_TRAIN}" "${N_VAL}" <<'EOF'
import sys, pandas as pd
src, n_train, n_val = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for split, n in (("train", n_train), ("val", n_val)):
    df = pd.read_csv(f"{src}/lsd_manifest_{split}.csv")
    df = df.sort_values(["group_id", "name"], na_position="last").head(n)
    df.to_csv(f"{split}.csv", index=False)
    print(f"{split}: {len(df)} rows, {df.group_id.nunique()} groups")
EOF

COMMON=(--csv train.csv --val-csv val.csv --loss censored_intra_assay_huber
        --epochs 2 --batch-size 8 --recycling-steps 1
        --early-stopping-patience 0 --device auto --no-plot-loss-curve --overwrite)

run_arm() {  # name, then the command
    local name="$1"; shift
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "gpu_${name}.txt" &
    local mon=$!
    local t0=$SECONDS
    "$@" 2>&1 | tee "train_${name}.log"
    kill "${mon}" 2>/dev/null || true
    echo "${name} wall_s=$((SECONDS - t0)) peak_gpu_MiB=$(sort -n "gpu_${name}.txt" | tail -1)" | tee -a timing.txt
}

run_arm lora_v2     boltz lora train --name lora_v2 "${COMMON[@]}" --rank 8 --alpha 16
run_arm lora_legacy boltz lora train --name lora_legacy "${COMMON[@]}" --rank 8 --alpha 16 \
                        --legacy-trainer --target-spec heads_pairformer
run_arm ft_v2       boltz finetune train --name ft_v2 "${COMMON[@]}" --learning-rate 1e-5 \
                        --target-spec affinity_heads_pairformer

# ── What actually trained, read from the artifacts ──────────────────────────
python - <<'EOF'
import collections, json, torch
def scope(k):
    top = k.split(".")[0]
    return f"{top}/heads" if ".affinity_heads." in k else f"{top}/stack" if ".pairformer_stack." in k else f"{top}/other"

for name in ("lora_v2", "lora_legacy"):
    meta = json.load(open(f"lora/{name}/meta.json"))
    run = meta["history"][-1]
    sd = torch.load(f"lora/{name}/adapter.pt", map_location="cpu", weights_only=False)["state_dict"]
    c = collections.defaultdict(lambda: [0, 0])
    for k, v in sd.items():
        if k.endswith("lora_B"):
            c[scope(k)][1] += 1; c[scope(k)][0] += int(v.abs().max() > 0)
    print(f"\n[{name}] semantics={run.get('trainer_semantics')} output={run.get('affinity_output')} "
          f"modules={run.get('target_modules')} sha={run.get('boltz_git_sha')} dirty={run.get('boltz_git_dirty')}")
    print("  adapted layers:", len(meta["adapted_layers"]))
    print("  layers with nonzero lora_B:", {s: f"{a}/{b}" for s, (a, b) in sorted(c.items())})
    print("  grad_coverage:", json.dumps(run.get("grad_coverage", {}).get("scopes")))
    print("  metrics:", run["metrics"])

base = torch.load("/home/limcaoco/turbo/limcaoco/.boltz/boltz2_aff.ckpt", map_location="cpu",
                  weights_only=False, mmap=True)["state_dict"]
w = torch.load("finetune/ft_v2/weights.pt", map_location="cpu", weights_only=False)
sd = w.get("state_dict", w)
c = collections.defaultdict(lambda: [0, 0])
for k, v in sd.items():
    c[scope(k)][1] += 1; c[scope(k)][0] += int(not torch.equal(v, base[k]))
run = json.load(open("finetune/ft_v2/meta.json"))["history"][-1]
print(f"\n[ft_v2] semantics={run.get('trainer_semantics')} output={run.get('affinity_output')}")
print("  tensors changed vs base:", {s: f"{a}/{b}" for s, (a, b) in sorted(c.items())})
print("  metrics:", run["metrics"])
print("\nEXPECTED: lora_v2 and ft_v2 change stack AND both modules; lora_legacy changes only affinity_module1/heads.")
EOF

# ── Rescore a few validation poses: vanilla vs lora_v2 ──────────────────────
python - "${N_RESCORE}" <<'EOF'
import os, sys, pandas as pd
n = int(sys.argv[1])
df = pd.read_csv("val.csv").head(n)
rows = []
for _, r in df.iterrows():
    stem = os.path.splitext(os.path.basename(r.structure))[0]
    dst = f"rescore_in/{stem}.pdb"
    if not os.path.exists(dst):
        os.symlink(r.structure, dst)
    rows.append({"name": stem, "smiles": r.ligand})
pd.DataFrame(rows).to_csv("rescore_smiles.csv", index=False)
print(f"rescoring {len(rows)} poses")
EOF
for tag in vanilla lora_v2; do
    extra=(); [[ "${tag}" == lora_v2 ]] && extra=(--use-lora lora_v2)
    boltz rescore batch --input-dir rescore_in --output "scores/${tag}.csv" --output-format csv \
        --device cuda --smiles-csv rescore_smiles.csv --msa-directory "${BOLTZ_MSA_CACHE_DIR}" \
        --recycling-steps 1 "${extra[@]}" --log-level INFO 2>&1 | tail -3
done
python - <<'EOF'
import pandas as pd
v, l = pd.read_csv("scores/vanilla.csv"), pd.read_csv("scores/lora_v2.csv")
for name, d in (("vanilla", v), ("lora_v2", l)):
    print(name, d.validation_status.value_counts().to_dict())
    bad = d[d.validation_status != "SUCCESS"]
    if len(bad):
        print("  failures:", bad.error_message.str.slice(0, 160).value_counts().to_dict())
m = v.merge(l, on="id", suffixes=("_v", "_l"))
m = m[(m.validation_status_v == "SUCCESS") & (m.validation_status_l == "SUCCESS")]
print("mean |lora_v2 - vanilla| on shared SUCCESS rows:", (m.affinity_pred_l - m.affinity_pred_v).abs().mean())
EOF

echo; cat timing.txt
echo "[$(date)] done: ${RUN_DIR}"
