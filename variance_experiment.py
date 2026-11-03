import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import torch
from utils.data import FEHashDataset
from utils.insightface import load_model, get_embedding


def create_projection(seed, D, d, projection_type):
    """
    Tái tạo Omega, r từ seed .
    """
    rng = np.random.default_rng(seed)

    if projection_type == "rff":
        Omega = rng.normal(0, 1, size=(D, d))
        r     = rng.uniform(0, 2 * np.pi, size=D)

    elif projection_type == "orf":
        if D % d != 0:
            raise ValueError(
                f"ORF yêu cầu D là bội số của d. D={D}, d={d}."
            )
        n_blocks = D // d
        blocks   = []
        for _ in range(n_blocks):
            G = rng.normal(0, 1, size=(d, d))
            Q, _ = np.linalg.qr(G)
            norm_samples = rng.normal(0, 1, size=(d, d))
            s = np.sqrt(np.sum(norm_samples ** 2, axis=1))
            blocks.append(np.diag(s) @ Q)
        Omega = np.vstack(blocks)
        r = rng.uniform(0, 2 * np.pi, size=D)

    else:
        raise ValueError("projection_type must be 'rff' or 'orf'")

    return Omega.astype(np.float64), r.astype(np.float64)

# ================Kernel estimate========================

def kernel_estimate_batch(X1, X2, Omega, r):
    """
    Tính K_hat(xi, xj) = phi(xi)^T phi(xj) cho tất cả N cặp.
    X1, X2: (N, d)
    Omega:  (D, d)
    r:      (D,)
    Return: (N,)
    """
    D = Omega.shape[0]
    Z1  = np.sqrt(2.0 / D) * np.cos(X1 @ Omega.T + r)  # (N, D)
    Z2  = np.sqrt(2.0 / D) * np.cos(X2 @ Omega.T + r)  # (N, D)
    return np.sum(Z1 * Z2, axis=1)                       # (N,)


def gaussian_kernel_true(X1, X2, sigma):
    """K(x,y) = exp(-||x-y||^2 / (2*sigma^2))"""
    diff = X1 - X2
    z2 = np.sum(diff**2, axis=1) / (sigma**2)
    return np.exp(-z2 / 2)



# ================Pair sampling==============================


def sample_pairs(embs, labels, n_pairs, pair_type, seed=0):
    """
    Trả về (X1, X2) shape (n_pairs, d) — cố định theo seed.
    pair_type: 'genuine' hoặc 'impostor'
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(labels)
    X1, X2  = [], []

    attempts = 0
    while len(X1) < n_pairs:
        attempts += 1
        if attempts > n_pairs * 100:
            raise RuntimeError(
                f"Không đủ cặp {pair_type} — kiểm tra lại dataset."
            )
        if pair_type == 'genuine':
            cls = rng.choice(classes)
            idx = np.where(labels == cls)[0]
            if len(idx) < 2:
                continue
            i, j = rng.choice(idx, size=2, replace=False)
        else:
            if len(classes) < 2:
                raise ValueError("Cần ít nhất 2 class để tạo impostor pairs.")
            c1, c2 = rng.choice(classes, size=2, replace=False)
            i = rng.choice(np.where(labels == c1)[0])
            j = rng.choice(np.where(labels == c2)[0])
        X1.append(embs[i])
        X2.append(embs[j])

    return np.array(X1[:n_pairs]), np.array(X2[:n_pairs])



def run_experiment(X1, X2, D, d, T, projection_type, base_seed=1000):
    """
    T lần sampling độc lập → tính K_hat → variance qua T lần.

    Return:
        K_trials:  (T, N) — kernel estimates
        var_pairs: (N,)   — empirical variance per pair
    """
    N   = len(X1)
    K_trials = np.zeros((T, N), dtype=np.float64)

    for t in range(T):
        seed = base_seed + t   # seed độc lập mỗi trial
        Omega, r = create_projection(seed, D, d, projection_type)
        K_trials[t] = kernel_estimate_batch(X1, X2, Omega, r)

    var_pairs = K_trials.var(axis=0, ddof=1)   # (N,) unbiased
    return K_trials, var_pairs



def theoretical_ratio(z, D, d):
    """
    Var(K_ORF) / Var(K_RFF) theo Yu et al. 2016:
      1 - (D-1)*exp(-z^2)*z^4 / (d*(1-exp(-z^2))^2)
    """
    ez2 = np.exp(-z**2)
    denom = d * (1 - ez2)**2
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    ratio = 1.0 - (D - 1) * ez2 * z**4 / denom
    return ratio



# ==========Plots==================================


def plot_results(results, z_gen, z_imp, D, d, sigma, save_path):
    """
    Vẽ 6 biểu đồ và lưu:
      - 1 file tổng hợp (save_path)
      - 6 file riêng lẻ (save_path stem + _plot1.png, ...)
    """
    colors = {'rff': '#2166AC', 'orf': '#1A9641'}
    base  = save_path.replace('.png', '')   # bỏ đuôi .png
 
    # ── Hàm lưu figure riêng ─────────────────────────────────────────────────
    def save_single(fig_s, suffix):
        path = f"{base}_{suffix}.png"
        fig_s.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig_s)
        print(f'  Saved → {path}')
 
    # ── 1. Variance distribution — genuine ───────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for m in ['rff', 'orf']:
        ax1.hist(results[m]['var_gen'], bins=50, alpha=0.6,
                 color=colors[m], density=True,
                 label=f"{m.upper()}  μ={results[m]['var_gen'].mean():.5f}")
    ax1.set_xlabel('Variance of kernel estimate')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Variance distribution — Genuine pairs (D={D})')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    save_single(fig1, 'var_genuine')
 
    # ── 2. Variance distribution — impostor ──────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for m in ['rff', 'orf']:
        ax2.hist(results[m]['var_imp'], bins=50, alpha=0.6,
                 color=colors[m], density=True,
                 label=f"{m.upper()}  μ={results[m]['var_imp'].mean():.5f}")
    ax2.set_xlabel('Variance of kernel estimate')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Variance distribution — Impostor pairs (D={D})')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save_single(fig2, 'var_impostor')
 
    # ── 3. K_hat distribution qua T trials ───────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    pair_idx  = 0
    for m in ['rff', 'orf']:
        ax3.hist(results[m]['K_gen'][:, pair_idx], bins=40, alpha=0.6,
                 color=colors[m], density=True, label=m.upper())
    ax3.axvline(np.exp(-z_gen[pair_idx]**2 / 2), color='red', lw=1.5,
                ls='--', label='True K')
    ax3.set_xlabel('K_hat value')
    ax3.set_ylabel('Density')
    ax3.set_title(f'K_hat distribution over {results["rff"]["T"]} trials\n'
                  f'(pair 0, genuine, z={z_gen[pair_idx]:.3f})')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    save_single(fig3, 'khat_genuine')
 
    # ── 4. Variance ratio per pair — genuine ─────────────────────────────────
    fig4, ax4   = plt.subplots(figsize=(6, 4))
    ratio_gen   = results['orf']['var_gen'] / (results['rff']['var_gen'] + 1e-12)
    ax4.hist(ratio_gen, bins=50, color='#762A83', alpha=0.7, density=True)
    ax4.axvline(1.0, color='red', lw=1.5, ls='--', label='ratio=1')
    ax4.axvline(ratio_gen.mean(), color='black', lw=1.5,
                ls='-', label=f'mean={ratio_gen.mean():.4f}')
    ax4.set_xlabel('Var(ORF) / Var(RFF)')
    ax4.set_ylabel('Density')
    ax4.set_title('Variance ratio per pair — Genuine')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    save_single(fig4, 'ratio_genuine')
 
    # ── 5. Variance ratio per pair — impostor ────────────────────────────────
    fig5, ax5   = plt.subplots(figsize=(6, 4))
    ratio_imp   = results['orf']['var_imp'] / (results['rff']['var_imp'] + 1e-12)
    ax5.hist(ratio_imp, bins=50, color='#D6604D', alpha=0.7, density=True)
    ax5.axvline(1.0, color='red', lw=1.5, ls='--', label='ratio=1')
    ax5.axvline(ratio_imp.mean(), color='black', lw=1.5,
                ls='-', label=f'mean={ratio_imp.mean():.4f}')
    ax5.set_xlabel('Var(ORF) / Var(RFF)')
    ax5.set_ylabel('Density')
    ax5.set_title('Variance ratio per pair — Impostor')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    save_single(fig5, 'ratio_impostor')
 
    # ── 6. Theory vs Empirical ────────────────────────────────────────────────
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    z_range   = np.linspace(0.01, 3.0, 300)
    th        = theoretical_ratio(z_range, D, d)
    ax6.plot(z_range, th, lw=2, color='#2166AC', label='Theory (Yu et al.)')
    ax6.axvline(z_gen.mean(), color='#1A9641', lw=1.5, ls='--',
                label=f'z̄ genuine={z_gen.mean():.3f}')
    ax6.axvline(z_imp.mean(), color='#D6604D', lw=1.5, ls='--',
                label=f'z̄ impostor={z_imp.mean():.3f}')
    ax6.scatter(z_gen.mean(),
                results['orf']['var_gen'].mean() /
                (results['rff']['var_gen'].mean() + 1e-12),
                color='#1A9641', zorder=5, s=80, label='Empirical (gen)')
    ax6.scatter(z_imp.mean(),
                results['orf']['var_imp'].mean() /
                (results['rff']['var_imp'].mean() + 1e-12),
                color='#D6604D', zorder=5, s=80, label='Empirical (imp)')
    ax6.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax6.set_xlabel('z = ||x - y|| / σ')
    ax6.set_ylabel('Var(ORF) / Var(RFF)')
    ax6.set_title('Comparison of Theory and Empirical ratio')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    save_single(fig6, 'theory_vs_empirical')
 
    # ── Tổng hợp 6 plot ───────────────────────────────────────────────────────
    fig, gs_ = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.42, wspace=0.35)
 
    def _copy_ax(src_path, ax_dst, title):
        img = plt.imread(src_path)
        ax_dst.imshow(img)
        ax_dst.axis('off')
 
    for i, suffix in enumerate([
        'var_genuine', 'var_impostor', 'khat_genuine',
        'ratio_genuine', 'ratio_impostor', 'theory_vs_empirical'
    ]):
        ax = fig.axes[i]
        img = plt.imread(f"{base}_{suffix}.png")
        ax.imshow(img)
        ax.axis('off')
 
    fig.suptitle(
        f'RFF and ORF Kernel Approximation Variance\n'
        f'D={D}, d={d}, T={results["rff"]["T"]} trials, σ={sigma}',
        fontsize=13, fontweight='bold'
    )
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Combined → {save_path}')




def main():
    parser = argparse.ArgumentParser('Variance RFF vs ORF')
    parser.add_argument('--data_root',   required=True,
                        help='Dataset root (e.g. data/cfpw_200)')
    parser.add_argument('--model_name',  default='buffalo_sc',
                        help='InsightFace model name')
    parser.add_argument('--D',           type=int, default=1024)
    parser.add_argument('--T',           type=int, default=500,
                        help='Number of independent projection seeds')
    parser.add_argument('--n_pairs',     type=int, default=200)
    parser.add_argument('--sigma',       type=float, default=1.0)
    parser.add_argument('--out_dir',     default='variance_results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load dataset + model ─────────────────────────────────────────────────
    print(f'\n[1] Loading dataset and model ...')

    ctx_id  = 0 if torch.cuda.is_available() else -1
    model   = load_model(args.model_name, ctx_id=ctx_id)
    dataset = FEHashDataset(args.data_root)
    labels  = dataset.get_all_labels()
    print(f'    Users: {len(labels)}')

    # ── Extract embeddings ────────────────────────────────────────────────────
    print(f'[2] Extracting embeddings ...')
    emb_list   = []
    label_list = []

    for user_label in labels:
        imgs = dataset.get_images_of_user(user_label)
        for img_path in imgs:
            try:
                emb = get_embedding(img_path, model)   # (d,) L2-normalized
                emb_list.append(emb)
                label_list.append(user_label)
            except Exception as e:
                print(f'    [SKIP] {img_path}: {e}')

    embs   = np.array(emb_list, dtype=np.float64)   # (N, d)
    labels = np.array(label_list)                    # (N,) string labels
    d      = embs.shape[1]

    # L2-normalize (InsightFace đã normalize nhưng đảm bảo chắc chắn)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.clip(norms, 1e-10, None)

    print(f'    Embeddings: {embs.shape}')
    print(f'    Classes   : {np.unique(labels).shape[0]}')
    print(f'    Norm check: {np.linalg.norm(embs[0]):.6f}  (should be 1.0)')

    if args.D % d != 0:
        raise ValueError(
            f'ORF yêu cầu D % d == 0. D={args.D}, d={d}. '
            f'Chọn D là bội số của {d}.'
        )

    # ── Sample pairs — cố định ───────────────────────────────────────────────
    print(f'\n[3] Sampling {args.n_pairs} genuine + {args.n_pairs} impostor pairs ...')
    X1_gen, X2_gen = sample_pairs(embs, labels, args.n_pairs, 'genuine',  seed=0)
    X1_imp, X2_imp = sample_pairs(embs, labels, args.n_pairs, 'impostor', seed=1)

    z_gen = np.linalg.norm(X1_gen - X2_gen, axis=1) / args.sigma
    z_imp = np.linalg.norm(X1_imp - X2_imp, axis=1) / args.sigma

    K_true_gen = gaussian_kernel_true(X1_gen, X2_gen, args.sigma)
    K_true_imp = gaussian_kernel_true(X1_imp, X2_imp, args.sigma)

    print(f'    Genuine  z: mean={z_gen.mean():.4f}  std={z_gen.std():.4f}')
    print(f'    Impostor z: mean={z_imp.mean():.4f}  std={z_imp.std():.4f}')
    print(f'    K_true genuine  mean: {K_true_gen.mean():.4f}')
    print(f'    K_true impostor mean: {K_true_imp.mean():.4f}')

    # ── Run experiment ────────────────────────────────────────────────────────
    results = {}
    for method in ['rff', 'orf']:
        print(f'\n[4] Running {method.upper()} — T={args.T} trials ...')

        K_gen, var_gen = run_experiment(
            X1_gen, X2_gen, args.D, d, args.T, method, base_seed=1000)
        print(f'    Genuine  done.')

        K_imp, var_imp = run_experiment(
            X1_imp, X2_imp, args.D, d, args.T, method, base_seed=2000)
        print(f'    Impostor done.')

        results[method] = {
            'var_gen': var_gen,
            'var_imp': var_imp,
            'K_gen':   K_gen,
            'K_imp':   K_imp,
            'T':       args.T,
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'RESULTS  (D={args.D}, d={d}, T={args.T}, σ={args.sigma})')
    print(f'{"="*65}')

    for pair_type, key in [('Genuine', 'gen'), ('Impostor', 'imp')]:
        print(f'\n  [{pair_type} pairs]')
        for m in ['rff', 'orf']:
            v = results[m][f'var_{key}']
            print(f'  {m.upper():4s}  mean={v.mean():.6f}  '
                  f'median={np.median(v):.6f}  std={v.std():.6f}')

        ratio_mean   = (results['orf'][f'var_{key}'].mean() /
                        (results['rff'][f'var_{key}'].mean() + 1e-12))
        ratio_median = (np.median(results['orf'][f'var_{key}']) /
                        (np.median(results['rff'][f'var_{key}']) + 1e-12))
        z_mean       = z_gen.mean() if key == 'gen' else z_imp.mean()
        th_ratio     = theoretical_ratio(z_mean, args.D, d)

        print(f'  Variance Ratio (mean)   : {ratio_mean:.4f}  '
              f'{"< 1: ORF lower" if ratio_mean < 1 else ">= 1: ORF not lower"}')
        print(f'  Variance Ratio (median) : {ratio_median:.4f}')
        print(f'  Theoretical ratio       : {th_ratio:.4f}  (z̄={z_mean:.4f})')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    rows = []
    for pair_type, key, z_arr in [
        ('genuine',  'gen', z_gen),
        ('impostor', 'imp', z_imp),
    ]:
        var_rff = results['rff'][f'var_{key}']
        var_orf = results['orf'][f'var_{key}']
        ratio   = var_orf / (var_rff + 1e-12)
        z_mean  = z_arr.mean()
        rows.append({
            'timestamp':        ts,
            'D':                args.D,
            'd':                d,
            'T':                args.T,
            'sigma':            args.sigma,
            'pair_type':        pair_type,
            'z_mean':           f'{z_mean:.4f}',
            'var_rff_mean':     f'{var_rff.mean():.8f}',
            'var_rff_std':     f'{var_rff.std():.8f}', 
            'var_rff_median':   f'{np.median(var_rff):.8f}',
            'var_orf_mean':     f'{var_orf.mean():.8f}',
            'var_orf_std':     f'{var_orf.std():.8f}',
            'var_orf_median':   f'{np.median(var_orf):.8f}',
            'ratio_mean':       f'{ratio.mean():.6f}',
            'ratio_std':       f'{ratio.std():.6f}',
            'ratio_median':     f'{np.median(ratio):.6f}',
            'theory_ratio':     f'{theoretical_ratio(z_mean, args.D, d):.6f}',
        })

    csv_path = os.path.join(args.out_dir, f'variance_results_{ts}.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'\n  CSV saved → {csv_path}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.out_dir, f'variance_plot_{ts}.png')
    plot_results(results, z_gen, z_imp, args.D, d, args.sigma, plot_path)

    print('\n[DONE]')


if __name__ == '__main__':
    main()
