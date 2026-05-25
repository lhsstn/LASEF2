"""
Analysis Figure: Scatter Plot comparing LLM-generated skeleton Δ vs Translated skeleton Δ
This produces a figure visually distinct from the Experiment heatmaps (Fig. 2),
showing whether skeleton-language effects persist when generation quality is controlled.

Points near the diagonal → effect persists regardless of quality (surface form matters)
Points far from diagonal → effect driven by quality differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['font.family'] = 'Times New Roman'
# matplotlib.rcParams['mathtext.fontset'] = 'stix'

# ============================================================
# 1. Load Data
# ============================================================
PRJ_PATH = "/home/work/mlp/hslim/LASEF2"

# LLM-generated skeleton delta (greedy, single rollout) — from Section 4.5 Fig.2
df_gen = pd.read_csv(
    f"{PRJ_PATH}/dataframe/single_rollout/Exist_Qwen2.5-7B-Instruct_single_pure_delta_filtered.csv",
    index_col=0
)

# Translated skeleton delta (translation ablation) — from Section 5 Analysis
df_trans = pd.read_csv(
    f"{PRJ_PATH}/dataframe/translated_skeleton_solved/Exist_Qwen2.5-7B-Instruct_pure_delta_filtered.csv",
    index_col=0
)

SKELETON_LANGUAGES = ["zh", "es", "ru", "ko", "th"]

LANGUAGE_NAMES = {
    "ta": "Tamil", "am": "Amharic", "si": "Sinhala", "gu": "Gujarati",
    "ne": "Nepali", "ky": "Kyrgyz", "eu": "Basque", "hy": "Armenian",
    "kk": "Kazakh", "lo": "Lao", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "mt": "Maltese", "or": "Odia", "pa": "Punjabi",
    "ps": "Pashto", "sd": "Sindhi", "ug": "Uyghur",
}

# ============================================================
# 2. Prepare scatter data
# ============================================================
records = []
for skel_lang in SKELETON_LANGUAGES:
    for cot_lang in df_gen.index:
        if cot_lang in df_trans.index:
            x_val = df_gen.loc[cot_lang, skel_lang]
            y_val = df_trans.loc[cot_lang, skel_lang]
            if pd.notna(x_val) and pd.notna(y_val):
                records.append({
                    'target': cot_lang,
                    'skeleton': skel_lang,
                    'gen_delta': x_val,
                    'trans_delta': y_val,
                    'label': f"{cot_lang}-{skel_lang}"
                })

df_scatter = pd.DataFrame(records)

# ============================================================
# 3. Define highlight groups (key pairs discussed in the paper)
# ============================================================
STABLE_PAIRS = [('hy', 'ko'), ('lo', 'th'), ('eu', 'ko')]
MALTESE_PAIRS = [('mt', 'ru'), ('mt', 'es'), ('mt', 'th'), ('mt', 'ko')]
ASYMMETRIC_PAIRS = [('mn', 'ru'), ('ug', 'zh')]

def get_category(row):
    pair = (row['target'], row['skeleton'])
    if pair in STABLE_PAIRS:
        return 'Stable cross-lingual'
    elif pair in MALTESE_PAIRS:
        return 'Maltese (eval-dependent)'
    elif pair in ASYMMETRIC_PAIRS:
        return 'Asymmetric negative'
    else:
        return 'Other'

df_scatter['category'] = df_scatter.apply(get_category, axis=1)

# ============================================================
# 4. Plot — focus on key pairs, background as density cloud
# ============================================================
fig, ax = plt.subplots(figsize=(11, 8))

# --- Background: show "Other" as a very subtle density cloud ---
other = df_scatter[df_scatter['category'] == 'Other']
ax.scatter(
    other['gen_delta'], other['trans_delta'],
    c='#D0D0D0', marker='o', s=45, alpha=0.25, zorder=1,
    label=f'Other pairs (n={len(other)})', edgecolors='none'
)

# --- Highlighted pairs ---
cat_styles = {
    'Stable cross-lingual':     {'color': '#2166AC', 'marker': 's', 's': 220, 'alpha': 0.95, 'zorder': 3},
    'Maltese (eval-dependent)': {'color': '#E8862A', 'marker': 'D', 's': 180, 'alpha': 0.95, 'zorder': 3},
    'Asymmetric negative':      {'color': '#B2182B', 'marker': '^', 's': 220, 'alpha': 0.95, 'zorder': 3},
}

for cat, style in cat_styles.items():
    subset = df_scatter[df_scatter['category'] == cat]
    ax.scatter(
        subset['gen_delta'], subset['trans_delta'],
        c=style['color'], marker=style['marker'], s=style['s'],
        alpha=style['alpha'], zorder=style['zorder'],
        label=cat, edgecolors='black', linewidths=1.2
    )

# --- Labels for highlighted points ---
highlight_pairs = STABLE_PAIRS + MALTESE_PAIRS + ASYMMETRIC_PAIRS
label_offsets = {
    'lo-th':  (-48, -10),
    'hy-ko':  (-48, 8),
    'eu-ko':  (-52, 8),
    'mt-ru':  (10, 8),
    'mt-es':  (10, 8),
    'mt-ko':  (-54, 8),
    'mt-th':  (10, -12),
    'mn-ru':  (10, 8),
    'ug-zh':  (10, -10),
}

for _, row in df_scatter.iterrows():
    pair = (row['target'], row['skeleton'])
    if pair in highlight_pairs:
        label_text = f"{row['target'].upper()}-{row['skeleton'].upper()}"
        offset = label_offsets.get(row['label'], (6, 6))
        ax.annotate(
            label_text,
            (row['gen_delta'], row['trans_delta']),
            textcoords="offset points",
            xytext=offset,
            fontsize=9.5, fontweight='bold',
            color=cat_styles[row['category']]['color'],
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5, boxstyle='round,pad=0.2')
        )

# --- Reference lines ---
lims = [-12, 12]
ax.plot(lims, lims, 'k--', alpha=0.25, linewidth=1, zorder=0)  # diagonal y=x
ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4, zorder=0)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)

# --- Quadrant labels (only the two informative ones) ---
ax.text(8.5, 8.5, 'Effect persists\n(surface form)', fontsize=9.5, ha='center', va='center',
        color='#555555', fontstyle='italic', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
ax.text(-8.5, -8.5, 'Negative in\nboth conditions', fontsize=9.5, ha='center', va='center',
        color='#555555', fontstyle='italic', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

# --- Axis labels and formatting ---
ax.set_xlabel('LLM-Generated Skeleton Δ (%)', fontsize=13)
ax.set_ylabel('Translated Skeleton Δ (quality-controlled) (%)', fontsize=13)
ax.set_title('Translation Ablation: Disentangling Quality from Surface Form', fontsize=14.5, fontweight='bold', pad=12)

ax.legend(
    loc='upper left',
    fontsize=10,
    framealpha=0.9,
    borderpad=0.8,
    labelspacing=1.0,
    handletextpad=1.0,
    markerscale=0.6
)
# ax.set_aspect('equal')
ax.grid(True, alpha=0.15)

plt.tight_layout()

# Save
save_path = f"{PRJ_PATH}/2026_May_Skeleton/Figures/Images/analysis_scatter_ablation.pdf"
plt.savefig(save_path, dpi=500, bbox_inches='tight')
print(f"✅ Scatter plot saved to: {save_path}")

save_path_png = save_path.replace('.pdf', '.png')
plt.savefig(save_path_png, dpi=500, bbox_inches='tight')
print(f"✅ PNG saved to: {save_path_png}")

plt.show()
