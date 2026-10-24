#!/usr/bin/env python3
"""
Generate publication-quality figures for the routing-message spectrum paper.
Table 1 (results_table.tex) and the spectrum figures of Section 6 and
Appendix D, from the 14-checkpoint measurement JSONs (4 families).

    python src/figures/plot_figures.py                 # shipped data/table1/
    python src/figures/plot_figures.py --results_dir results/measure/table1

Fonts: Times New Roman if installed, otherwise DejaVu Serif. The published
figures were rendered with the DejaVu Serif fallback, so with Times New Roman
installed the glyphs and text widths (not the data) differ from the paper.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

FAMILY_COLORS = {
    'Qwen3': '#1f77b4',
    'Llama3': '#d62728',
    'Gemma2': '#2ca02c',
    'Mistral': '#ff7f0e',
}
FAMILY_MARKERS = {
    'Qwen3': 'o',
    'Llama3': 's',
    'Gemma2': '^',
    'Mistral': 'D',
}

MODEL_ORDER = [
    'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b', 'qwen3-14b', 'qwen3-32b',
    'llama3.2-1b', 'llama3.2-3b', 'llama3-8b', 'llama3-70b',
    'gemma2-2b', 'gemma2-9b',
    'mistral-7b-v03', 'mistral-nemo-12b',
]

PARAM_SIZES = {
    'qwen3-0.6b': 0.6, 'qwen3-1.7b': 1.7, 'qwen3-4b': 4, 'qwen3-8b': 8,
    'qwen3-14b': 14, 'qwen3-32b': 32,
    'llama3.2-1b': 1.2, 'llama3.2-3b': 3.2, 'llama3-8b': 8, 'llama3-70b': 70,
    'gemma2-2b': 2.5, 'gemma2-9b': 9.2,
    'mistral-7b-v03': 7.2, 'mistral-nemo-12b': 12,
}

SHORT_NAMES = {
    'qwen3-0.6b': 'Q3-0.6B', 'qwen3-1.7b': 'Q3-1.7B', 'qwen3-4b': 'Q3-4B',
    'qwen3-8b': 'Q3-8B', 'qwen3-14b': 'Q3-14B', 'qwen3-32b': 'Q3-32B',
    'llama3.2-1b': 'L3-1B', 'llama3.2-3b': 'L3-3B', 'llama3-8b': 'L3-8B', 'llama3-70b': 'L3-70B',
    'gemma2-2b': 'G2-2B', 'gemma2-9b': 'G2-9B',
    'mistral-7b-v03': 'Mi-7B', 'mistral-nemo-12b': 'Mi-12B',
}


def get_family(model_key):
    if 'qwen' in model_key: return 'Qwen3'
    if 'llama' in model_key: return 'Llama3'
    if 'gemma' in model_key: return 'Gemma2'
    if 'mistral' in model_key: return 'Mistral'
    return 'Other'


def load_all_results(results_dir):
    all_data = {}
    for f in sorted(Path(results_dir).glob('results*.json')):
        with open(f) as fh:
            data = json.load(fh)
        all_data.update(data)
    # Filter out broken models
    return {k: v for k, v in all_data.items() if k in MODEL_ORDER}


def get_stats(d):
    layers = d['layers']
    r_effs = [layers[l]['r_eff'] for l in layers]
    t1s = [layers[l]['tail_energy'].get('1', 0) for l in layers]
    t4s = [layers[l]['tail_energy'].get('4', 0) for l in layers]
    t8s = [layers[l]['tail_energy'].get('8', 0) for l in layers]
    return {
        'r_eff_median': np.median(r_effs), 'r_eff_min': np.min(r_effs), 'r_eff_max': np.max(r_effs),
        't1_median': np.median(t1s), 't4_median': np.median(t4s), 't8_median': np.median(t8s),
        'r_effs': r_effs, 't1s': t1s, 't4s': t4s, 't8s': t8s,
    }


# ── fig1_reff_per_layer (paper Figure 1, top right) ──────────────
def fig1_reff_per_layer(data, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for mk in MODEL_ORDER:
        if mk not in data: continue
        d = data[mk]
        layers = sorted(d['layers'].keys(), key=int)
        r_eff = [d['layers'][l]['r_eff'] for l in layers]
        x = np.linspace(0, 1, len(r_eff))
        fam = get_family(mk)
        ax.plot(x, r_eff, label=SHORT_NAMES.get(mk, mk),
                color=FAMILY_COLORS[fam], linewidth=0.9, alpha=0.75,
                marker=FAMILY_MARKERS[fam], markersize=1.5, markevery=max(1, len(r_eff)//8))
    ax.set_xlabel('Relative layer position')
    ax.set_ylabel('$r_{\\mathrm{eff}}$')
    ax.legend(loc='upper right', ncol=3, framealpha=0.9, columnspacing=0.8, handletextpad=0.3)
    ax.set_ylim(bottom=0)
    fig.savefig(outdir / 'fig1_reff_per_layer.pdf')
    plt.close(fig)
    print("Saved fig1")


# ── fig2_bar_chart (paper Figure 1, top left) ────────────────────
def fig2_bar_chart(data, outdir):
    models = [m for m in MODEL_ORDER if m in data]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    x = np.arange(len(models))
    w = 0.25
    q_heads = [data[m]['num_query_heads'] for m in models]
    kv_heads = [data[m]['num_kv_heads'] for m in models]
    r_effs = [get_stats(data[m])['r_eff_median'] for m in models]
    ax.bar(x - w, q_heads, w, label='$H_q$', color='#4c72b0', alpha=0.8)
    ax.bar(x, kv_heads, w, label='$H_{kv}$', color='#55a868', alpha=0.8)
    ax.bar(x + w, r_effs, w, label='$r_{\\mathrm{eff}}$', color='#c44e52', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(m, m) for m in models], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Count / Rank')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / 'fig2_bar_chart.pdf')
    plt.close(fig)
    print("Saved fig2")


# ── fig3_scale_scatter (paper Figure 1, bottom left) ─────────────
def fig3_scale_scatter(data, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))
    models = [m for m in MODEL_ORDER if m in data]
    for m in models:
        fam = get_family(m)
        s = get_stats(data[m])
        ax1.scatter(PARAM_SIZES[m], s['r_eff_median'], c=FAMILY_COLORS[fam],
                    marker=FAMILY_MARKERS[fam], s=50, zorder=5, edgecolors='black', linewidth=0.4)
        ax1.annotate(SHORT_NAMES.get(m, m), (PARAM_SIZES[m], s['r_eff_median']),
                     fontsize=5.5, textcoords='offset points', xytext=(3, 3))
        ratio = s['r_eff_median'] / data[m]['num_query_heads']
        ax2.scatter(PARAM_SIZES[m], ratio, c=FAMILY_COLORS[fam],
                    marker=FAMILY_MARKERS[fam], s=50, zorder=5, edgecolors='black', linewidth=0.4)
        ax2.annotate(SHORT_NAMES.get(m, m), (PARAM_SIZES[m], ratio),
                     fontsize=5.5, textcoords='offset points', xytext=(3, 3))
    ax1.set_xscale('log'); ax1.set_xlabel('Parameters (B)'); ax1.set_ylabel('Median $r_{\\mathrm{eff}}$')
    ax2.set_xscale('log'); ax2.set_xlabel('Parameters (B)'); ax2.set_ylabel('$r_{\\mathrm{eff}} / H_q$')
    # Legend by family
    for fam in FAMILY_COLORS:
        ax1.scatter([], [], c=FAMILY_COLORS[fam], marker=FAMILY_MARKERS[fam], label=fam, s=40)
    ax1.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(outdir / 'fig3_scale_scatter.pdf')
    plt.close(fig)
    print("Saved fig3")


# ── fig4_heatmap (paper Figure 2, right) ─────────────────────────
def fig4_heatmap(data, outdir):
    R_vals = ['1', '2', '4', '8', '16']
    models = [m for m in MODEL_ORDER if m in data]
    matrix = np.zeros((len(models), len(R_vals)))
    for i, m in enumerate(models):
        for j, R in enumerate(R_vals):
            vals = [data[m]['layers'][l]['tail_energy'].get(R, 0) for l in data[m]['layers']]
            matrix[i, j] = np.median(vals)
    fig, ax = plt.subplots(figsize=(4, 4.5))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.4)
    ax.set_xticks(range(len(R_vals)))
    ax.set_xticklabels([f'$R={r}$' for r in R_vals])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([SHORT_NAMES.get(m, m) for m in models], fontsize=7)
    for i in range(len(models)):
        for j in range(len(R_vals)):
            v = matrix[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if v > 0.2 else 'black')
    ax.set_xlabel('Route budget $R$')
    fig.colorbar(im, ax=ax, shrink=0.7, label='Median $T_R$')
    fig.tight_layout()
    fig.savefig(outdir / 'fig4_heatmap.pdf')
    plt.close(fig)
    print("Saved fig4")


# ── fig5_sv_grid (paper Figure 2, left) ──────────────────────────
def fig5_sv_grid(data, outdir):
    models = [m for m in MODEL_ORDER if m in data]
    n = len(models)
    ncols, nrows = 4, (n + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 1.8 * nrows), sharex=False)
    axes = axes.flatten()
    for idx, mk in enumerate(models):
        ax = axes[idx]
        d = data[mk]
        layers = sorted(d['layers'].keys(), key=int)
        nl = len(layers)
        reps = [layers[0], layers[nl//3], layers[2*nl//3], layers[-1]]
        for l in reps:
            sigma = np.array(d['layers'][l]['sigma'])
            if len(sigma) == 0 or sigma[0] == 0: continue
            sigma_n = sigma / sigma[0]
            ax.plot(range(1, len(sigma_n)+1), sigma_n, linewidth=0.8, alpha=0.7, label=f'L{l}')
        ax.set_title(f"{SHORT_NAMES.get(mk, mk)} ({d['num_query_heads']}Q/{d['num_kv_heads']}KV)", fontsize=7)
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1.5)
        ax.legend(fontsize=4.5, loc='upper right', handlelength=1)
        if idx % ncols == 0: ax.set_ylabel('$\\sigma_k/\\sigma_1$', fontsize=8)
        if idx >= (nrows - 1) * ncols: ax.set_xlabel('$k$', fontsize=8)
    for idx in range(n, nrows * ncols):
        axes[idx].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / 'fig5_sv_grid.pdf')
    plt.close(fig)
    print("Saved fig5")


# ── fig6_family_scaling (paper Figure 1, bottom right) ───────────
# (family, panel title, bar colour) of the published panel
FAMILY_BARS = [
    ('Qwen3', 'Qwen3', '#2563eb'),
    ('Llama3', 'Llama 3', '#dc2626'),
    ('Gemma2', 'Gemma 2', '#059669'),
    ('Mistral', 'Mistral', '#d97706'),
]


def fig6_family_scaling(data, outdir):
    """Median r_eff per checkpoint as bars grouped by family, ordered by size,
    with the H_kv = 8 reference line (Figure 1, bottom-right panel). Size,
    colours and spacing follow the published panel."""
    fig, ax = plt.subplots(figsize=(6.96, 2.195))
    ticks, ticklabels, groups = [], [], []
    start = 0.0
    for fam, title, color in FAMILY_BARS:
        models = [m for m in MODEL_ORDER if m in data and get_family(m) == fam]
        if not models:
            continue
        x = start + np.arange(len(models))
        med = [get_stats(data[m])['r_eff_median'] for m in models]
        bars = ax.bar(x, med, width=0.7, color=color, alpha=0.9,
                      edgecolor='white', linewidth=0.5, zorder=3)
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in med], padding=2.65,
                     fontsize=7, color='#374151')
        ticks += list(x)
        ticklabels += [f'{PARAM_SIZES[m]:g}B' for m in models]
        groups.append((title, x))
        start = x[-1] + 1.8  # gap between families
    for (_, left), (_, right) in zip(groups, groups[1:]):
        ax.axvline((left[-1] + right[0]) / 2, color='#d1d5db', linestyle='--',
                   linewidth=0.5, zorder=0)
    ax.axhline(8, color='#9ca3af', linestyle=':', linewidth=0.6, alpha=0.7, zorder=0)
    ax.text(ticks[-1] + 0.5, 8.2, '$H_{kv}=8$', fontsize=7, color='#6b7280',
            ha='left', va='bottom')
    for title, x in groups:
        ax.annotate(title, xy=(x.mean(), 1), xycoords=('data', 'axes fraction'),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=7.5)
    ax.set_ylim(0, 14.5)
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(which='major', width=0.6)
    ax.set_ylabel('Median $r_{\\mathrm{eff}}$', fontsize=9)
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(outdir / 'fig6_family_scaling.pdf', pad_inches=0.1)
    plt.close(fig)
    print("Saved fig6")


# ── fig7_reff_range (paper Figure 3, Appendix D) ─────────────────
def fig7_reff_range(data, outdir):
    models = [m for m in MODEL_ORDER if m in data]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for i, m in enumerate(models):
        s = get_stats(data[m])
        fam = get_family(m)
        ax.barh(i, s['r_eff_max'] - s['r_eff_min'], left=s['r_eff_min'],
                color=FAMILY_COLORS[fam], alpha=0.3, height=0.6)
        ax.plot(s['r_eff_median'], i, marker='|', color=FAMILY_COLORS[fam],
                markersize=12, markeredgewidth=2)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([SHORT_NAMES.get(m, m) for m in models], fontsize=7)
    ax.set_xlabel('$r_{\\mathrm{eff}}$ (min / median / max across layers)')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(outdir / 'fig7_reff_range.pdf')
    plt.close(fig)
    print("Saved fig7")


# ── LaTeX table ──────────────────────────────────────────────────
def generate_table(data, outdir):
    models = [m for m in MODEL_ORDER if m in data]
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Family & Model & Params & $H_q$ & $H_{kv}$ & Layers & $r_{\mathrm{eff}}$ & $T_1$ & $T_4$ & $T_8$ \\",
        r"\midrule",
    ]
    for m in models:
        d = data[m]
        s = get_stats(d)
        fam = get_family(m)
        lines.append(
            f"{fam} & {SHORT_NAMES.get(m, m)} & {PARAM_SIZES.get(m, '?')}B & "
            f"{d['num_query_heads']} & {d['num_kv_heads']} & {len(d['layers'])} & "
            f"{s['r_eff_median']:.1f} & {s['t1_median']:.3f} & {s['t4_median']:.3f} & {s['t8_median']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / 'results_table.tex').write_text("\n".join(lines))
    print("Saved table")
    print("\n".join(lines))


def main():
    import argparse
    root = Path(__file__).resolve().parents[2]  # repository root (src/figures/ -> root)
    ap = argparse.ArgumentParser(description='Table 1 and spectrum figures from measurement JSONs.')
    ap.add_argument('--results_dir', default=str(root / 'data' / 'table1'),
                    help='directory with results*.json (default: shipped data/table1/)')
    ap.add_argument('--outdir', default=str(root / 'results' / 'figures'))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_all_results(results_dir)
    print(f"Loaded {len(data)} models: {list(data.keys())}\n")

    fig1_reff_per_layer(data, outdir)
    fig2_bar_chart(data, outdir)
    fig3_scale_scatter(data, outdir)
    fig4_heatmap(data, outdir)
    fig5_sv_grid(data, outdir)
    fig6_family_scaling(data, outdir)
    fig7_reff_range(data, outdir)
    generate_table(data, outdir)
    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
