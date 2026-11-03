import os
import re
import numpy as np
import csv
import hashlib
import time
from datetime import datetime

from utils.insightface import get_embedding, load_model
from utils.data import FEHashDataset, build_test_dataset
from utils.utils_io import load_pickle_zst
from AVET import AVET
from FEHash import FEHash


#=============================================================================

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


#=====================================================================================

def run_fixed_one_seed(dataset, save_folder, model,
                       use_avet=False,
                       n_pos=8, n_neg=100,
                       random_seed=42,
                       measure_time=False):
    """
    Hai loại tổng thời gian per-query:

    Total_Time      : embed + avet_transform + hash
                      (giả định Omega,r,R,A,B
                      đã có sẵn trong bộ nhớ)

    Total_Time_seed : Total_Time
                      + tái tạo 1 lần AVET (R,A,B từ seed)
                      + tái tạo K lần FEHash (Omega,r từ seed)
                      (kịch bản seed-based: không lưu ma trận, phải
                      tái tạo từ seed cho mỗi query)
    """
    test_data = build_test_dataset(
        dataset, n_pos=n_pos, n_neg=n_neg,
        random_seed=random_seed
    )

    genuine_accept  = genuine_total  = 0
    impostor_accept = impostor_total = 0

    all_times      = []   # Total_Time
    all_seed_times = []   # Total_Time_seed
    embed_times    = []
    avet_times     = []
    hash_times     = []

    for user_id, pairs in test_data.items():
        user_dir = os.path.join(save_folder, user_id)
        if not os.path.isdir(user_dir):
            continue

        with open(os.path.join(user_dir,
                               "bitstring_sha3_256.txt")) as f:
            enroll_sha = f.read().strip()

        K = len([f for f in os.listdir(user_dir)
                 if f.startswith("bit_")])

        # ── Load ──────────────────
        fehash_bits = [
            FEHash.load(os.path.join(user_dir, f"bit_{i}.pkl.zst"))
            for i in range(K)
        ]
        avet = _load_avet_for_user(user_id, save_folder) \
               if use_avet else None

        # ── Đo thời gian tái tạo từ seed (1 lần/user) ────────────────
        # Mô phỏng kịch bản mỗi query phải tái tạo từ seed
        if measure_time:
            # K lần tái tạo Omega, r (FEHash)
            t0 = time.perf_counter()
            for i in range(K):
                FEHash.load(os.path.join(user_dir, f"bit_{i}.pkl.zst"))
            load_fehash_ms = (time.perf_counter() - t0) * 1000

            # 1 lần tái tạo R, A, B (AVET)
            if use_avet:
                t0 = time.perf_counter()
                _load_avet_for_user(user_id, save_folder)
                load_avet_ms = (time.perf_counter() - t0) * 1000
            else:
                load_avet_ms = 0.0

            regen_overhead_ms = load_fehash_ms + load_avet_ms
        else:
            regen_overhead_ms = 0.0

        def get_sha(img_path):
            t_total = time.perf_counter()

            t0  = time.perf_counter()
            emb = get_embedding(img_path, model).astype(np.float64)
            if measure_time:
                embed_times.append(time.perf_counter() - t0)

            if use_avet:
                t0  = time.perf_counter()
                emb = avet.enroll(emb, save=False)
                emb = emb / np.linalg.norm(emb)
                if measure_time:
                    avet_times.append(time.perf_counter() - t0)

            t0   = time.perf_counter()
            bits = "".join(str(fe.hash(emb)) for fe in fehash_bits)
            sha  = hashlib.sha3_256(bits.encode()).hexdigest()
            if measure_time:
                hash_times.append(time.perf_counter() - t0)
                query_ms = (time.perf_counter() - t_total) * 1000
                all_times.append(query_ms)
                all_seed_times.append(query_ms + regen_overhead_ms)

            return sha

        for img_path in pairs["pos"]:
            genuine_total += 1
            if get_sha(img_path) == enroll_sha:
                genuine_accept += 1

        for img_path in pairs["neg"]:
            impostor_total += 1
            if get_sha(img_path) == enroll_sha:
                impostor_accept += 1

    GAR = genuine_accept  / genuine_total  if genuine_total  else 0.0
    FAR = impostor_accept / impostor_total if impostor_total else 0.0

    timing = None
    if measure_time and all_times:
        timing = {
            # Total_Time
            "total_ms":          np.mean(all_times),
            "total_ms_std":      np.std(all_times),
            # Total_Time_seed — thêm overhead tái tạo từ seed
            "total_seed_ms":     np.mean(all_seed_times),
            "total_seed_ms_std": np.std(all_seed_times),
            # Breakdown
            "embed_ms":          np.mean(embed_times) * 1000,
            "embed_ms_std":      np.std(embed_times)  * 1000,
            "avet_ms":           np.mean(avet_times)  * 1000 if avet_times else 0.0,
            "avet_ms_std":       np.std(avet_times)   * 1000 if avet_times else 0.0,
            "hash_ms":           np.mean(hash_times)  * 1000,
            "hash_ms_std":       np.std(hash_times)   * 1000,
            # Overhead tái tạo (per user/query)
            "load_fehash_ms":    load_fehash_ms   if measure_time else 0.0,
            "load_avet_ms":      load_avet_ms     if measure_time else 0.0,
            "regen_overhead_ms": regen_overhead_ms if measure_time else 0.0,
        }

    return GAR, FAR, timing


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fixed(dataset, method_dataset_dir, model,
                   use_avet=False, n_pos=8, n_neg=100):

    seed_dirs = _find_seed_dirs(method_dataset_dir)
    if not seed_dirs:
        raise RuntimeError(
            f"Không tìm thấy thư mục seed_* trong: {method_dataset_dir}"
        )

    print(f"  Tìm thấy {len(seed_dirs)} seed: "
          f"{[s for s, _ in seed_dirs]}")

    GARs, FARs  = [], []
    all_timings = []

    for seed_val, seed_path in seed_dirs:
        GAR, FAR, t = run_fixed_one_seed(
            dataset      = dataset,
            save_folder  = seed_path,
            model        = model,
            use_avet     = use_avet,
            n_pos        = n_pos,
            n_neg        = n_neg,
            random_seed  = seed_val,
            measure_time = True,
        )
        GARs.append(GAR)
        FARs.append(FAR)
        if t:
            all_timings.append(t)

        msg = (f"  seed_{seed_val} | "
               f"GAR={GAR*100:.4f}%  FAR={FAR*100:.4f}%")
        if t:
            msg += (f"  Total={t['total_ms']:.3f}ms"
                    f"  Total_seed={t['total_seed_ms']:.3f}ms")
        print(msg)

    gar_mean, gar_std = np.mean(GARs), np.std(GARs)
    far_mean, far_std = np.mean(FARs), np.std(FARs)

    print(f"\n  → GAR = {gar_mean*100:.4f} ± {gar_std*100:.4f} %")
    print(f"  → FAR = {far_mean*100:.4f} ± {far_std*100:.4f} %")

    timing_summary = None
    if all_timings:
        def _mean(k): return float(np.mean([t[k] for t in all_timings]))
        def _std(k):  return float(np.std( [t[k] for t in all_timings]))

        timing_summary = {
            "total_ms":          _mean("total_ms"),
            "total_ms_std":      _std("total_ms"),
            "total_seed_ms":     _mean("total_seed_ms"),
            "total_seed_ms_std": _std("total_seed_ms"),
            "embed_ms":          _mean("embed_ms"),
            "embed_ms_std":      _std("embed_ms"),
            "avet_ms":           _mean("avet_ms"),
            "avet_ms_std":       _std("avet_ms"),
            "hash_ms":           _mean("hash_ms"),
            "hash_ms_std":       _std("hash_ms"),
            "load_fehash_ms":    _mean("load_fehash_ms"),
            "load_avet_ms":      _mean("load_avet_ms"),
            "regen_overhead_ms": _mean("regen_overhead_ms"),
        }
        t = timing_summary
        print(f"\n  [TIMING mean±std over {len(all_timings)} seeds]")
        print(f"  Total_Time      : "
              f"{t['total_ms']:.3f} ± {t['total_ms_std']:.3f} ms"
              f"  (matrices pre-loaded)")
        print(f"  Total_Time_seed : "
              f"{t['total_seed_ms']:.3f} ± {t['total_seed_ms_std']:.3f} ms"
              f"  (regen from seed each query)")
        print(f"  --- breakdown ---")
        print(f"  embed              : {t['embed_ms']:.3f} ± {t['embed_ms_std']:.3f} ms")
        if use_avet:
            print(f"  AVET transform     : {t['avet_ms']:.3f} ± {t['avet_ms_std']:.3f} ms")
        print(f"  FEHash + hash      : {t['hash_ms']:.3f} ± {t['hash_ms_std']:.3f} ms")
        print(f"  regen Omega,r (×K) : {t['load_fehash_ms']:.3f} ms")
        if use_avet:
            print(f"  regen R,A,B        : {t['load_avet_ms']:.3f} ms")
        print(f"  regen total        : {t['regen_overhead_ms']:.3f} ms")

    return {
        "GAR_mean": gar_mean, "GAR_std": gar_std,
        "FAR_mean": far_mean, "FAR_std": far_std,
        "GARs":     GARs,     "FARs":    FARs,
        "n_seeds":  len(seed_dirs),
        "seeds":    [s for s, _ in seed_dirs],
        "timing":   timing_summary,
    }


# ====================================================================================

def save_result(result_dir, method_label, dataset_name, result):
    os.makedirs(result_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Per-seed CSV ───────────────────────────────────────────────
    seed_csv    = os.path.join(result_dir, "fixed_per_seed.csv")
    seed_exists = os.path.isfile(seed_csv)
    with open(seed_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not seed_exists:
            writer.writerow([
                "timestamp", "method", "dataset",
                "seed", "GAR%", "FAR%",
            ])
        for seed_val, GAR, FAR in zip(
            result["seeds"], result["GARs"], result["FARs"]
        ):
            writer.writerow([
                ts, method_label, dataset_name,
                seed_val,
                f"{GAR*100:.4f}",
                f"{FAR*100:.4f}",
            ])

    # ── 2. Summary CSV ────────────────────────────────────────────────
    sum_csv    = os.path.join(result_dir, "fixed_results.csv")
    sum_exists = os.path.isfile(sum_csv)
    t = result["timing"] or {}
    with open(sum_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not sum_exists:
            writer.writerow([
                "timestamp", "method", "dataset",
                "GAR_mean%",         "GAR_std%",
                "FAR_mean%",         "FAR_std%",
                "n_seeds",           "seeds",
                
                "Total_Time_ms",     "Total_Time_ms_std",
                
                "Total_Time_seed_ms","Total_Time_seed_ms_std",
                # breakdown
                "embed_ms",          "embed_ms_std",
                "avet_ms",           "avet_ms_std",
                "hash_ms",           "hash_ms_std",
                "regen_fehash_ms",
                "regen_avet_ms",
                "regen_total_ms",
            ])
        writer.writerow([
            ts, method_label, dataset_name,
            f"{result['GAR_mean']*100:.4f}",
            f"{result['GAR_std']*100:.4f}",
            f"{result['FAR_mean']*100:.4f}",
            f"{result['FAR_std']*100:.4f}",
            result["n_seeds"],
            str(result["seeds"]),
            f"{t.get('total_ms',          0):.3f}",
            f"{t.get('total_ms_std',      0):.3f}",
            f"{t.get('total_seed_ms',     0):.3f}",
            f"{t.get('total_seed_ms_std', 0):.3f}",
            f"{t.get('embed_ms',          0):.3f}",
            f"{t.get('embed_ms_std',      0):.3f}",
            f"{t.get('avet_ms',           0):.3f}",
            f"{t.get('avet_ms_std',       0):.3f}",
            f"{t.get('hash_ms',           0):.3f}",
            f"{t.get('hash_ms_std',       0):.3f}",
            f"{t.get('load_fehash_ms',    0):.3f}",
            f"{t.get('load_avet_ms',      0):.3f}",
            f"{t.get('regen_overhead_ms', 0):.3f}",
        ])

    print(f"  Per-seed → {seed_csv}")
    print(f"  Summary  → {sum_csv}")


# ========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fixed Matching Evaluation")
    parser.add_argument("--data_root",          required=True)
    parser.add_argument("--method_dataset_dir", required=True)
    parser.add_argument("--result_dir",         required=True)
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
    print(f"n_pos   : {args.n_pos}  |  n_neg : {args.n_neg}")
    print(f"{'='*55}")

    result = evaluate_fixed(
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
