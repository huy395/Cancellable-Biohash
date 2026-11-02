"""
plot_roc_tunable.py

Đọc cấu trúc:
    eval_results/tunable/{method}/{dataset}/roc_data.npz

Tạo 1 hình ROC per dataset, vẽ tất cả methods trên cùng hình.
Lưu tại:
    eval_results/tunable/roc_{dataset}.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


METHOD_STYLES = {
    "rff_D1024_C1e-3": {
        "label": "RFF, $D=1024$",
        "color": "#2166AC",
        "ls": "--",
        "lw": 2.8,
        "order": 0,
    },
    "rff_D4096_C1e-3": {
        "label": "RFF, $D=4096$",
        "color": "#D6604D",
        "ls": "-.",
        "lw": 2.8,
        "order": 1,
    },
    "orf_D1024_C1e-3": {
        "label": "ORF, $D=1024$ (proposed)",
        "color": "#1A9641",
        "ls": "-",
        "lw": 3.2,
        "order": 2,
    },
    "bi_avet": {
        "label": "Bi-AVET (baseline)",
        "color": "#762A83",
        "ls": ":",
        "lw": 3.2,
        "order": 3,
    },
}

# Fallback style cho method không có trong dict trên
DEFAULT_STYLE = {
    "color": "#888888",
    "ls":    "-",
    "lw":    1.5,
}

COMMON_FPR = np.linspace(0, 1, 300)


# ─────────────────────────────────────────────────────────────────────────────

def load_roc_data(npz_path):
    """Load roc_data.npz, tái tạo fpr_list/tpr_list từ key fpr_seed{N}."""
    data  = np.load(npz_path, allow_pickle=True)
    seeds = data["seeds"].tolist()

    fpr_list = [data[f"fpr_seed{s}"] for s in seeds]
    tpr_list = [data[f"tpr_seed{s}"] for s in seeds]

    return {
        "fpr_list": fpr_list,
        "tpr_list": tpr_list,
        "auc_mean": float(data["auc_mean"]),
        "auc_std":  float(data["auc_std"]),
        "eer_mean": float(data["eer_mean"]),
        "eer_std":  float(data["eer_std"]),
        "seeds":    seeds,
    }


def scan_eval_root(eval_root):
    """
    Quét eval_root/{method}/{dataset}/roc_data.npz
    Trả về dict: { dataset: { method: npz_path } }
    """
    result = defaultdict(dict)

    for method_name in sorted(os.listdir(eval_root)):
        method_dir = os.path.join(eval_root, method_name)
        if not os.path.isdir(method_dir):
            continue

        for dataset_name in sorted(os.listdir(method_dir)):
            dataset_dir = os.path.join(method_dir, dataset_name)
            npz_path    = os.path.join(dataset_dir, "roc_data.npz")

            if os.path.isfile(npz_path):
                result[dataset_name][method_name] = npz_path
            else:
                print(f"  [SKIP] {method_name}/{dataset_name} — no roc_data.npz")

    return result


def plot_roc_one_dataset(dataset_name, methods_data, save_path):

    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    handles = []
    found = 0

    # chỉ vẽ 3 method
    methods_to_plot = [
        "rff_D1024_C1e-3",
        "rff_D4096_C1e-3",
        "orf_D1024_C1e-3",
    ]

    def sort_key(method_name):
        return METHOD_STYLES.get(method_name, {}).get("order", 99)

    for method_name in sorted(methods_data.keys(), key=sort_key):

        if method_name not in methods_to_plot:
            continue

        npz_path = methods_data[method_name]
        style = METHOD_STYLES[method_name]

        try:
            d = load_roc_data(npz_path)
        except Exception as e:
            print(f"[ERROR] {method_name}: {e}")
            continue

        tpr_interp = np.array([
            np.interp(COMMON_FPR, fpr, tpr)
            for fpr, tpr in zip(d["fpr_list"], d["tpr_list"])
        ])

        tpr_mean = tpr_interp.mean(axis=0)
        tpr_std  = tpr_interp.std(axis=0)

        if "1024" in method_name:
            marker = "o"
        elif "4096" in method_name:
            marker = "s"
        else:
            marker = "^"

        ax.plot(
            COMMON_FPR,
            tpr_mean,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=1.0,
            marker=marker,
            markevery=20,
            markersize=1.0,
        )

        ax.fill_between(
            COMMON_FPR,
            np.clip(tpr_mean - tpr_std, 0, 1),
            np.clip(tpr_mean + tpr_std, 0, 1),
            color=style["color"],
            alpha=0.08,
        )

        legend_label = (
            f"{style['label']}\n"
            f"AUC={d['auc_mean']:.4f}±{d['auc_std']:.4f}\n"
            f"EER={d['eer_mean']*100:.2f}±{d['eer_std']*100:.2f}%"
        )

        handles.append(
            mlines.Line2D(
                [],
                [],
                color=style["color"],
                linestyle=style["ls"],
                linewidth=1.0,
                marker=marker,
                markersize=1.0,
                label=legend_label,
            )
        )

        found += 1

    if found == 0:
        plt.close()
        return

    # zoom theo dataset
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0.6, 1.01)

    ax.set_title(
        f"ROC Curve — {dataset_name}",
        fontsize=18,
    )

    ax.set_xlabel(
        "False Accept Rate (FAR)",
        fontsize=14,
    )

    ax.set_ylabel(
        "Genuine Accept Rate (GAR)",
        fontsize=14,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    ax.legend(
        handles=handles,
        fontsize=9,
        loc="lower right",
        framealpha=0.95,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved -> {save_path}")


# ===========================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot ROC Tunable")
    parser.add_argument(
        "--eval_root",
        default="eval_results_1/tunable",
        help="Thư mục chứa {method}/{dataset}/roc_data.npz",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Thư mục lưu ảnh ROC. Mặc định = eval_root",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or args.eval_root
    os.makedirs(save_dir, exist_ok=True)

    # Quét tất cả method/dataset
    all_data = scan_eval_root(args.eval_root)

    if not all_data:
        print(f"[ERROR] Không tìm thấy roc_data.npz nào trong {args.eval_root}")
        exit(1)

    print(f"\nTìm thấy {len(all_data)} dataset(s): {list(all_data.keys())}")

    for dataset_name, methods_data in sorted(all_data.items()):
        print(f"\n[PLOT] {dataset_name} — {len(methods_data)} method(s): "
              f"{list(methods_data.keys())}")

        save_path = os.path.join(save_dir, f"roc_{dataset_name}.png")
        plot_roc_one_dataset(dataset_name, methods_data, save_path)

    print("\n[ALL DONE]")
