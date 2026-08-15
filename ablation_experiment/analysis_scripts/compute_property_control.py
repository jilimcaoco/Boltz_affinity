#!/usr/bin/env python3
"""Task 5 — ligand-property control.

Why this matters under the new hypothesis
------------------------------------------
The hypothesis is that the affinity head shortcuts on receptor/ligand
*identity* via z_trunk/s_inputs rather than genuinely reading the predicted
3D complex. But "shortcuts via z_trunk/s_inputs" and "shortcuts via 2D
ligand properties such as size" are *different* shortcuts and must not be
conflated -- if an ablation effect just tracks molecular size or logP, that
is still a shortcut finding, but a different one, and has to be reported as
such rather than folded into the trunk/s_inputs attribution from Task 2.

Two independent controls
-------------------------
1. **Property-residual control.** Per (receptor, experiment), regress the
   raw affinity_pred_value on 4 continuous ligand descriptors (heavy-atom
   count, cLogP, TPSA, formal charge) and recompute logAUC on the
   *residuals*. If an ablation's effect survives on residuals, it isn't
   explained by these 4 properties alone.
2. **2D ceiling + AVE bias.** Per receptor, a ligand-only ECFP4 classifier
   (radius 2, 2048 bits, L2-logistic regression, 5-fold CV) with *no*
   structural/model input at all gives logAUC_2D -- the performance
   achievable by memorizing 2D chemical similarity between actives and
   decoys alone. AVE bias (Wallach & Heifets 2018) measures how much a
   train/test split's own nearest-neighbor similarity structure inflates
   that number, independent of any real signal. Every ablation effect can
   then be reported as **structure-attributable excess**:
   ``logAUC - logAUC_2D`` -- the fraction of performance that cannot be
   explained by 2D similarity to the training actives/decoys.

This is also how you distinguish a genuine bias_only (Task 0's v(∅)) signal
from the tie-bug artifact: if bias_only's residual/structure-attributable
numbers are still near zero after this control, that corroborates Task 0's
fix rather than hinting at a second confound.

Data
----
Ligand structures come from the same DOCK3.8 MOL2 pose files
run_feature_ablation.py uses (SMILES inferred the same way, via
``boltz.affinity_rescoring.smiles_inference``), joined against
``results/ablation/feature_ablation_results.csv`` for raw scores by
``ligand_name``. Requires the full boltz environment (rdkit + the
featurization stack) -- this cannot run standalone.

Convention: ZINC-prefixed ligand_name = decoy; everything else = active.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logauc_utils import compute_logauc, label_actives_decoys  # noqa: E402

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ABLATION_CSV = BASE_DIR / "results" / "ablation" / "feature_ablation_results.csv"
POSES_DIR = BASE_DIR / "DOCK3.8_poses"
DEFAULT_DUDEZ_INPUTS = Path(
    "/home/limcaoco/turbo/limcaoco/boltz_benchmark/input_files/DUDEZ_benchmark"
)
OUT_DIR = BASE_DIR / "analysis_data"

ECFP4_RADIUS = 2
ECFP4_NBITS = 2048
N_CV_FOLDS = 5


def is_decoy(name: str) -> bool:
    return str(name).startswith("ZINC")


def clean_ligand_name(raw) -> str:
    return re.sub(r"\s+none$", "", str(raw)).strip()


# ── RDKit descriptors / fingerprints (pure functions, no boltz dependency) ──

def compute_ligand_properties(smiles: str) -> Optional[Dict[str, float]]:
    """Heavy-atom count, cLogP (Crippen), TPSA, formal charge."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "heavy_atom_count": float(mol.GetNumHeavyAtoms()),
        "clogp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
    }


def compute_ecfp4(smiles: str) -> Optional[np.ndarray]:
    """ECFP4 (Morgan, radius=2) as a 2048-bit numpy array of 0/1 int8."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP4_RADIUS, fpSize=ECFP4_NBITS)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((ECFP4_NBITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ── property-residual control (Task 5, part 1) ──────────────────────────────

def compute_residual_logauc(
    scores: Dict[str, float],
    properties: Dict[str, Dict[str, float]],
    lig_set: set,
    dec_set: set,
    min_compounds: int = 8,
) -> Tuple[float, float, int]:
    """Regress ``scores`` on [heavy_atom_count, clogp, tpsa, formal_charge]
    for compounds with both a score and computed properties, and return
    ``(residual_logauc, r_squared, n_compounds)``. Residual logAUC uses the
    same lower-is-better convention as the raw scores (residual = actual -
    predicted, so a compound that ranks better than its properties alone
    would predict still has a lower residual)."""
    from sklearn.linear_model import LinearRegression

    common = sorted(set(scores) & set(properties))
    if len(common) < min_compounds:
        return float("nan"), float("nan"), len(common)

    X = np.array([[properties[c]["heavy_atom_count"], properties[c]["clogp"],
                    properties[c]["tpsa"], properties[c]["formal_charge"]] for c in common])
    y = np.array([scores[c] for c in common])

    reg = LinearRegression().fit(X, y)
    residuals = y - reg.predict(X)
    r_squared = float(reg.score(X, y))

    ranked = sorted(zip(common, residuals), key=lambda x: x[1])
    lig_present = lig_set & set(common)
    dec_present = dec_set & set(common)
    if not lig_present or not dec_present:
        return float("nan"), r_squared, len(common)

    logauc = compute_logauc(ranked, lig_present, dec_present)
    return logauc, r_squared, len(common)


# ── 2D-only classifier + AVE bias (Task 5, part 2) ──────────────────────────

def _tanimoto_nn_mean(query_fps: np.ndarray, ref_fps: np.ndarray) -> float:
    """Mean nearest-neighbor Tanimoto similarity from each row of
    ``query_fps`` to its closest row in ``ref_fps`` (binary vectors)."""
    if len(query_fps) == 0 or len(ref_fps) == 0:
        return float("nan")
    inter = query_fps.astype(np.float64) @ ref_fps.astype(np.float64).T
    q_sum = query_fps.sum(axis=1, keepdims=True).astype(np.float64)
    r_sum = ref_fps.sum(axis=1, keepdims=True).astype(np.float64).T
    union = q_sum + r_sum - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return float(sim.max(axis=1).mean())


def ave_bias(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """AVE bias (Wallach & Heifets 2018): how much a train/test split's own
    nearest-neighbor similarity structure -- independent of any learned
    signal -- inflates apparent classifier performance. Sum, over actives
    and decoys, of (within-class NN similarity to train) minus (cross-class
    NN similarity to train). Near zero = unbiased split; large positive =
    a memorization-only classifier would still look good on this split."""
    A_tr, D_tr = X_train[y_train == 1], X_train[y_train == 0]
    A_te, D_te = X_test[y_test == 1], X_test[y_test == 0]
    term_actives = _tanimoto_nn_mean(A_te, A_tr) - _tanimoto_nn_mean(A_te, D_tr)
    term_decoys = _tanimoto_nn_mean(D_te, D_tr) - _tanimoto_nn_mean(D_te, A_tr)
    parts = [t for t in (term_actives, term_decoys) if not np.isnan(t)]
    return float(sum(parts)) if parts else float("nan")


def compute_2d_logauc_and_ave(
    fingerprints: Dict[str, np.ndarray],
    lig_set: set,
    dec_set: set,
    n_splits: int = N_CV_FOLDS,
    seed: int = 42,
) -> Dict[str, float]:
    """5-fold stratified CV L2-logistic regression on ECFP4 fingerprints
    alone (no structural/model input). Returns out-of-fold logAUC_2D and the
    mean AVE bias across folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    names = sorted((lig_set | dec_set) & set(fingerprints))
    if len(names) < n_splits * 2:
        return {"logauc_2d": float("nan"), "ave_bias": float("nan"), "n_compounds": len(names)}

    X = np.stack([fingerprints[n] for n in names])
    y = np.array([1 if n in lig_set else 0 for n in names])
    if len(set(y.tolist())) < 2:
        return {"logauc_2d": float("nan"), "ave_bias": float("nan"), "n_compounds": len(names)}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_scores: Dict[str, float] = {}
    ave_biases: List[float] = []

    for train_idx, test_idx in skf.split(X, y):
        y_train = y[train_idx]
        if len(set(y_train.tolist())) < 2:
            continue
        clf = LogisticRegression(penalty="l2", max_iter=2000).fit(X[train_idx], y_train)
        proba = clf.predict_proba(X[test_idx])[:, 1]
        for i, p in zip(test_idx, proba):
            oof_scores[names[i]] = float(p)
        ave_biases.append(ave_bias(X[train_idx], y_train, X[test_idx], y[test_idx]))

    if not oof_scores:
        return {"logauc_2d": float("nan"), "ave_bias": float("nan"), "n_compounds": len(names)}

    # higher predicted probability of "active" = better; do_roc convention
    # is lower-is-better, so rank by negated probability.
    ranked = sorted(((n, -p) for n, p in oof_scores.items()), key=lambda x: x[1])
    scored_names = set(oof_scores)
    logauc_2d = compute_logauc(ranked, lig_set & scored_names, dec_set & scored_names)
    return {
        "logauc_2d": logauc_2d,
        "ave_bias": float(np.mean(ave_biases)) if ave_biases else float("nan"),
        "n_compounds": len(names),
    }


# ── data loading (needs the full boltz env; not covered by unit tests) ─────

def load_ligand_smiles_for_receptor(poses_path: Path) -> Dict[str, str]:
    """Extract {ligand_name: smiles} from a DOCK3.8 MOL2 pose file, using
    the same SMILES-inference path run_feature_ablation.py uses.

    Only for the MOL2 flow. The DUDEZ flow uses
    :func:`load_manifest_smiles_for_receptor` instead -- its SMILES are
    authoritative and need no RDKit bond perception.
    """
    from boltz.affinity_rescoring.mol2_parser import MOL2Parser
    from boltz.affinity_rescoring.smiles_inference import infer_smiles_from_atoms

    parser = MOL2Parser()
    ligands = parser.extract_ligands_with_names(poses_path)
    out = {}
    for ligand in ligands:
        try:
            smiles = infer_smiles_from_atoms(ligand.atoms)
        except Exception:
            smiles = None
        if smiles:
            out[ligand.name] = smiles
    return out


def load_manifest_smiles_for_receptor(manifest_path: Path) -> Dict[str, str]:
    """Extract {compound_ID: smiles} from a DUDEZ manifest CSV
    (``<RECEPTOR>_combined_ids.csv``, columns SMILES / compound_ID /
    is_binder). Pure stdlib csv -- no boltz or RDKit dependency, unlike the
    MOL2 path, because the SMILES are already authoritative."""
    import csv as _csv

    out: Dict[str, str] = {}
    with manifest_path.open() as fh:
        reader = _csv.DictReader(fh)
        fields = reader.fieldnames or []
        smiles_key = next((k for k in fields if k.strip().upper() == "SMILES"), None)
        cid_key = next((k for k in fields if k.strip().lower() == "compound_id"), None)
        if smiles_key is None or cid_key is None:
            raise ValueError(
                f"Cannot locate SMILES/compound_ID columns in {manifest_path}: got {fields}"
            )
        for row in reader:
            cid = (row.get(cid_key) or "").strip()
            sm = (row.get(smiles_key) or "").strip()
            if cid and sm:
                out[cid] = sm
    return out


def resolve_smiles_source(receptor: str, poses_dir: Path, manifest_dir: Optional[Path]):
    """Return ``(smiles_by_name, source)`` for one receptor, preferring the
    DUDEZ manifest when available and falling back to MOL2 poses.

    Both ablation runners feed the same analysis chain, so this control has
    to accept either input shape. ``(None, reason)`` when neither exists.
    """
    if manifest_dir is not None:
        manifest_path = manifest_dir / f"{receptor}_combined_ids.csv"
        if manifest_path.exists():
            return load_manifest_smiles_for_receptor(manifest_path), f"manifest:{manifest_path.name}"

    poses_path = poses_dir / f"{receptor}_poses.mol2"
    if poses_path.exists():
        return load_ligand_smiles_for_receptor(poses_path), f"mol2:{poses_path.name}"

    return None, (
        f"no SMILES source: neither {manifest_dir}/{receptor}_combined_ids.csv "
        f"(DUDEZ manifest) nor {poses_dir}/{receptor}_poses.mol2 (MOL2 poses) exists"
    )


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation-csv", type=Path, default=ABLATION_CSV)
    parser.add_argument("--poses-dir", type=Path, default=POSES_DIR,
                         help="MOL2 pose files (run_feature_ablation.py flow).")
    parser.add_argument("--dudez-inputs-root", type=Path, default=DEFAULT_DUDEZ_INPUTS,
                         help="Directory of <RECEPTOR>_combined_ids.csv manifests "
                              "(run_feature_ablation_dudez.py flow). Preferred over "
                              "--poses-dir when a manifest exists for the receptor.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.ablation_csv.exists():
        raise SystemExit(f"{args.ablation_csv} not found. Run run_feature_ablation.py first.")

    df = pd.read_csv(args.ablation_csv)
    df["ligand_id"] = df["ligand_name"].apply(clean_ligand_name)
    df = df[df["error"].isna() & df["affinity_pred_value"].notna()]

    receptors = sorted(df["receptor_id"].unique())
    logger.info(f"Receptors: {', '.join(receptors)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bias_rows = []
    residual_rows = []

    for receptor in receptors:
        smiles_by_name, source = resolve_smiles_source(
            receptor, args.poses_dir, args.dudez_inputs_root,
        )
        if smiles_by_name is None:
            logger.warning(f"[{receptor}] skipping -- {source}")
            continue
        logger.info(f"[{receptor}] SMILES source: {source} ({len(smiles_by_name)} compounds)")
        properties = {}
        fingerprints = {}
        for name, smiles in smiles_by_name.items():
            props = compute_ligand_properties(smiles)
            fp = compute_ecfp4(smiles)
            if props is not None:
                properties[name] = props
            if fp is not None:
                fingerprints[name] = fp

        df_rec = df[df["receptor_id"] == receptor]
        lig_set, dec_set, _label_source = label_actives_decoys(
            df_rec.drop_duplicates("ligand_id")
        )
        ratio = (len(lig_set) / len(dec_set)) if dec_set else float("nan")

        result = compute_2d_logauc_and_ave(fingerprints, lig_set, dec_set, seed=args.seed)
        bias_rows.append({
            "receptor": receptor,
            "logauc_2d": result["logauc_2d"],
            "ave_bias": result["ave_bias"],
            "n_compounds_with_fingerprint": result["n_compounds"],
            "n_actives": len(lig_set),
            "n_decoys": len(dec_set),
            "actives_decoys_ratio": ratio,
        })
        logger.info(f"[{receptor}] logAUC_2D={result['logauc_2d']:.2f} "
                    f"AVE_bias={result['ave_bias']:.3f} "
                    f"({len(lig_set)} actives : {len(dec_set)} decoys)")

        for experiment, df_exp in df_rec.groupby("experiment"):
            best = df_exp.groupby("ligand_id")["affinity_pred_value"].min()
            scores = best.to_dict()
            resid_logauc, r2, n_used = compute_residual_logauc(scores, properties, lig_set, dec_set)
            residual_rows.append({
                "receptor": receptor, "experiment": experiment,
                "residual_logauc": resid_logauc, "property_r_squared": r2,
                "n_compounds_used": n_used,
            })

    if not bias_rows:
        raise SystemExit(
            f"No receptor produced any ligand-property data -- every receptor in "
            f"{args.ablation_csv.name} was skipped, most likely because no matching "
            f"'<RECEPTOR>_poses.mol2' exists under {args.poses_dir}. This control needs "
            f"the ligand structures, not just the scores. Point --poses-dir at the "
            f"DOCK3.8 pose files and re-run."
        )

    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(args.output_dir / "receptor_bias_profile.csv", index=False)
    logger.info(f"Wrote {len(bias_df)} rows to receptor_bias_profile.csv")

    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(args.output_dir / "property_residual_logauc.csv", index=False)
    logger.info(f"Wrote {len(residual_df)} rows to property_residual_logauc.csv")

    # Join structure-attributable excess if the ablation bootstrap summary
    # is available (raw logAUC per (experiment, receptor)).
    ablation_summary_path = args.output_dir / "ablation_bootstrap_results" / "summary_per_receptor.csv"
    if ablation_summary_path.exists():
        summary_df = pd.read_csv(ablation_summary_path)
        joined = summary_df.merge(
            bias_df[["receptor", "logauc_2d", "ave_bias", "actives_decoys_ratio"]],
            on="receptor", how="left",
        ).merge(
            residual_df, on=["receptor", "experiment"], how="left",
        )
        joined["structure_attributable_excess"] = joined["point_estimate_logAUC"] - joined["logauc_2d"]
        out_path = args.output_dir / "ablation_with_property_control.csv"
        joined.to_csv(out_path, index=False)
        logger.info(f"Wrote {len(joined)} rows to {out_path.name} "
                    f"(raw logAUC, residual logAUC, and structure-attributable excess)")
    else:
        logger.warning(f"{ablation_summary_path} not found -- run compute_ablation_bootstrap.py "
                        f"first to get the joined structure-attributable-excess table.")


if __name__ == "__main__":
    main()
