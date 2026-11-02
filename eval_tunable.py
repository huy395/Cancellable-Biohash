import os
import re
import numpy as np
import csv
import hashlib
import torch
from datetime import datetime
from sklearn.metrics import roc_curve, auc as sk_auc

from utils.insightface import get_embeddings_with_aug, load_model
from utils.data import FEHashDataset, build_test_dataset
from utils.utils_io import load_pickle_zst
from AVET import AVET
from FEHash import FEHash


# =================================================================================

def _load_avet_for_user(user_id, save_folder):
    path   = os.path.join(save_folder, user_id, "avet_params.zst")
    params = load_pickle_zst(path)
    if params.get("_avet_version", 0) >= 8:
        return AVET(m=params["m"], n=params["n"],
                    user_seed=params["user_seed"],
                    save_dir=save_folder)
    avet = AVET(m=params["m"], n=params["n"],
                user_seed=None, save_dir=save_folder)
    avet.R = params["R"].astype(np.float64)
    avet.A = params["A"].astype(np.float64)
    avet.B = params["B"].astype(np.float64)
    return avet


def _find_seed_dirs(method_dataset_dir):
    entries = []
    for name in os.listdir(method_dataset_dir):
        full = os.path.join(method_dataset_dir, name)
        if os.path.isdir(full) and re.match(r"^seed_\d+$", name):
            entries.append((int(name.replace("seed_", "")), full))
    entries.sort(key=lambda x: x[0])
    return entries


# ===================================================================================

def run_tunable_one_seed(dataset, save_folder, model,
                         use_avet=False,
                         n_pos=8, n_neg=100,
                         random_seed=42):

    test_data = build_test_dataset(
        dataset, n_pos=n_pos, n_neg=n_neg,
        random_seed=random_seed
    )

    genuine_scores  = []
    impostor_scores = []
    img_counter     = 0

    for user_id, pairs in test_data.items():
        user_dir = os.path.join(save_folder, user_id)
        if not os.path.isdir(user_dir):
            continue

        with open(os.path.join(user_dir,
                               "bitstring_sha3_256.txt")) as f:
            enroll_sha = f.read().strip()

        K = len([f for f in os.listdir(user_dir)
                 if f.startswith("bit_")])
        fehash_bits = [
            FEHash.load(os.path.join(user_dir, f"bit_{i}.pkl.zst"))
            for i in range(K)
        ]
        avet = _load_avet_for_user(user_id, save_folder) \
               if use_avet else None

        def get_score(img_path, idx):
            np.random.seed(random_seed + idx)
            torch.manual_seed(random_seed + idx)
            embs = get_embeddings_with_aug(
                img_path, model).astype(np.float64)
            if use_avet:
                new_embs = []
                for emb in embs:
                    y = avet.enroll(emb, save=False)
                    new_embs.append(y / np.linalg.norm(y))
                embs = np.array(new_embs)
            match = sum(
                hashlib.sha3_256(
                    "".join(str(fe.hash(emb))
                            for fe in fehash_bits).encode()
                ).hexdigest() == enroll_sha
                for emb in embs
            )
            return match / len(embs)

        for img_path in pairs["pos"]:
            genuine_scores.append(get_score(img_path, img_counter))
            img_counter += 1
        for img_path in pairs["neg"]:
            impostor_scores.append(get_score(img_path, img_counter))
            img_counter += 1

    y_true  = [1]*len(genuine_scores) + [0]*len(impostor_scores)
    y_score = genuine_scores + impostor_scores

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc              = sk_auc(fpr, tpr)
    frr                  = 1 - tpr
    eer_idx              = np.argmin(np.abs(fpr - frr))
    eer                  = float((fpr[eer_idx] + frr[eer_idx]) / 2)
    eer_threshold        = float(thresholds[eer_idx])

    return {
        "fpr":           fpr,
        "tpr":           tpr,
        "thresholds":    thresholds,
        "auc":           roc_auc,
        "eer":           eer,
        "eer_threshold": eer_threshold,
        "GAR":           float(tpr[eer_idx]),
        "FAR":           float(fpr[eer_idx]),
    }


# ===============================================================================

def evaluate_tunable(dataset, method_dataset_dir, model,
                     use_avet=False, n_pos=8, n_neg=100):

    seed_dirs = _find_seed_dirs(method_dataset_dir)
    if not seed_dirs:
        raise RuntimeError(
            f"Không tìm thấy seed_* trong: {method_dataset_dir}"
        )
    print(f"  Seeds: {[s for s, _ in seed_dirs]}")

    all_fpr, all_tpr, all_thresholds = [], [], []
    GARs, FARs, EERs, AUCs          = [], [], [], []
    EER_thresholds                   = []
    seed_vals                        = []

    for seed_val, seed_path in seed_dirs:
        r = run_tunable_one_seed(
            dataset, seed_path, model,
            use_avet=use_avet,
            n_pos=n_pos, n_neg=n_neg,
            random_seed=seed_val,
        )
        all_fpr.append(r["fpr"])
        all_tpr.append(r["tpr"])
        all_thresholds.append(r["thresholds"])
        GARs.append(r["GAR"])
        FARs.append(r["FAR"])
        EERs.append(r["eer"])
        AUCs.append(r["auc"])
        EER_thresholds.append(r["eer_threshold"])
        seed_vals.append(seed_val)

        print(f"  seed_{seed_val} | "
              f"EER={r['eer']*100:.4f}%  "
              f"τ={r['eer_threshold']:.4f}  "
              f"AUC={r['auc']:.4f}  "
              f"GAR={r['GAR']*100:.4f}%  "
              f"FAR={r['FAR']*100:.4f}%")

    gar_mean, gar_std       = np.mean(GARs), np.std(GARs)
    far_mean, far_std       = np.mean(FARs), np.std(FARs)
    eer_mean, eer_std       = np.mean(EERs), np.std(EERs)
    auc_mean, auc_std       = np.mean(AUCs), np.std(AUCs)
    eer_thresh_mean         = np.mean(EER_thresholds)
    eer_thresh_std          = np.std(EER_thresholds)

    print(f"\n  → EER     = {eer_mean*100:.4f} ± {eer_std*100:.4f} %")
    print(f"  → τ (EER) = {eer_thresh_mean:.4f} ± {eer_thresh_std:.4f}")
    print(f"  → AUC     = {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"  → GAR     = {gar_mean*100:.4f} ± {gar_std*100:.4f} %")
    print(f"  → FAR     = {far_mean*100:.4f} ± {far_std*100:.4f} %")

    return {
        "fpr_list":           all_fpr,
        "tpr_list":           all_tpr,
        "thresholds_list":    all_thresholds,
        "GAR_mean": gar_mean, "GAR_std":  gar_std,
        "FAR_mean": far_mean, "FAR_std":  far_std,
        "EER_mean": eer_mean, "EER_std":  eer_std,
        "AUC_mean": auc_mean, "AUC_std":  auc_std,
        "EER_threshold_mean": eer_thresh_mean,
        "EER_threshold_std":  eer_thresh_std,
        "GARs": GARs, "FARs": FARs,
        "EERs": EERs, "AUCs": AUCs,
        "EER_thresholds": EER_thresholds,
        "seeds":   seed_vals,
        "n_seeds": len(seed_vals),
    }


# =====================================================================================

def save_result(result_dir, method_label, dataset_name, result):
    os.makedirs(result_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Per-seed CSV ───────────────────────────────────────────────
    seed_csv    = os.path.join(result_dir, "tunable_per_seed.csv")
    seed_exists = os.path.isfile(seed_csv)
    with open(seed_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not seed_exists:
            writer.writerow([
                "timestamp", "method", "dataset",
                "seed", "GAR%", "FAR%", "EER%", "AUC",
                "EER_threshold",
            ])
        for seed_val, GAR, FAR, EER, AUC, thresh in zip(
            result["seeds"],
            result["GARs"],    result["FARs"],
            result["EERs"],    result["AUCs"],
            result["EER_thresholds"],
        ):
            writer.writerow([
                ts, method_label, dataset_name,
                seed_val,
                f"{GAR*100:.4f}",
                f"{FAR*100:.4f}",
                f"{EER*100:.4f}",
                f"{AUC:.4f}",
                f"{thresh:.4f}",
            ])

    # ── 2. Summary mean±std CSV ───────────────────────────────────────
    sum_csv    = os.path.join(result_dir, "tunable_results.csv")
    sum_exists = os.path.isfile(sum_csv)
    with open(sum_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not sum_exists:
            writer.writerow([
                "timestamp", "method", "dataset",
                "GAR_mean%",     "GAR_std%",
                "FAR_mean%",     "FAR_std%",
                "EER_mean%",     "EER_std%",
                "AUC_mean",      "AUC_std",
                "EER_thresh_mean", "EER_thresh_std",
                "n_seeds", "seeds",
            ])
        writer.writerow([
            ts, method_label, dataset_name,
            f"{result['GAR_mean']*100:.4f}",
            f"{result['GAR_std']*100:.4f}",
            f"{result['FAR_mean']*100:.4f}",
            f"{result['FAR_std']*100:.4f}",
            f"{result['EER_mean']*100:.4f}",
            f"{result['EER_std']*100:.4f}",
            f"{result['AUC_mean']:.4f}",
            f"{result['AUC_std']:.4f}",
            f"{result['EER_threshold_mean']:.4f}",
            f"{result['EER_threshold_std']:.4f}",
            result["n_seeds"],
            str(result["seeds"]),
        ])

    # ── 3. ROC data .npz ─────────────────────────────────────────────
    npz_path  = os.path.join(result_dir, "roc_data.npz")
    save_dict = {
        "method_label":      np.array(method_label),
        "dataset_name":      np.array(dataset_name),
        "auc_mean":          np.float64(result["AUC_mean"]),
        "auc_std":           np.float64(result["AUC_std"]),
        "eer_mean":          np.float64(result["EER_mean"]),
        "eer_std":           np.float64(result["EER_std"]),
        "eer_threshold_mean":np.float64(result["EER_threshold_mean"]),
        "eer_threshold_std": np.float64(result["EER_threshold_std"]),
        "eer_thresholds":    np.array(result["EER_thresholds"]),
        "n_seeds":           np.int32(result["n_seeds"]),
        "seeds":             np.array(result["seeds"]),
    }
    for seed_val, fpr, tpr, thresh in zip(
        result["seeds"],
        result["fpr_list"],
        result["tpr_list"],
        result["thresholds_list"],
    ):
        save_dict[f"fpr_seed{seed_val}"]        = fpr
        save_dict[f"tpr_seed{seed_val}"]        = tpr
        save_dict[f"thresholds_seed{seed_val}"] = thresh

    np.savez(npz_path, **save_dict)

    print(f"  Per-seed → {seed_csv}")
    print(f"  Summary  → {sum_csv}")
    print(f"  ROC data → {npz_path}")


# ====================================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Tunable Matching Evaluation")
    parser.add_argument("--data_root",          required=True)
    parser.add_argument("--method_dataset_dir", required=True,
                        help="e.g. templates/orf_D1024_C1e-2/cfpw")
    parser.add_argument("--result_dir",         required=True,
                        help="e.g. eval_results/tunable/orf_D1024_C1e-2/cfpw")
    parser.add_argument("--method_label",       required=True)
    parser.add_argument("--use_avet",           action="store_true")
    parser.add_argument("--n_pos",              type=int, default=8)
    parser.add_argument("--n_neg",              type=int, default=100)
    args = parser.parse_args()

    model        = load_model()
    dataset      = FEHashDataset(args.data_root)
    dataset_name = os.path.basename(args.data_root.rstrip("/\\"))

    print(f"\n{'='*55}")
    print(f"Method  : {args.method_label}")
    print(f"Dataset : {dataset_name}")
    print(f"Dir     : {args.method_dataset_dir}")
    print(f"n_pos={args.n_pos}  n_neg={args.n_neg}")
    print(f"{'='*55}")

    result = evaluate_tunable(
        dataset            = dataset,
        method_dataset_dir = args.method_dataset_dir,
        model              = model,
        use_avet           = args.use_avet,
        n_pos              = args.n_pos,
        n_neg              = args.n_neg,
    )

    save_result(
        result_dir   = args.result_dir,
        method_label = args.method_label,
        dataset_name = dataset_name,
        result       = result,
    )
