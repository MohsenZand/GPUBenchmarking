import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


def build_comprehensive_visuals():
    os.makedirs('figures', exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    plt.rcParams.update({'axes.titleweight': 'bold', 'axes.titlesize': 13,
                         'figure.titlesize': 18, 'figure.titleweight': 'bold'})

    CLUSTER_COLORS = {'Lenovo': '#E74C3C', 'Schmidt': '#2980B9'}

    # CPU threads: Lenovo B200=112, B300=172 (full-node, from vLLM OMP warning);
    #              Schmidt = Slurm allocation (--cpus-per-task=64, --mem=500G in run_bench.sh)
    SPECS = {
        'B200': {
            'Lenovo':  'D: 595.58 | C: 13.2\n8× B200 SXM | 179 GB HBM3e\n112 CPU threads | 1000 W TDP',
            'Schmidt': 'D: 595.71 | C: 13.2\n8× B200 SXM | 179 GB HBM3e\n64 CPU | 500 GB RAM | 1000 W TDP',
        },
        'B300': {
            'Lenovo':  'D: 595.71 | C: 13.2\n8× B300 SXM6 AC | 269 GB HBM3e\n172 CPU threads | 1100 W TDP',
            'Schmidt': 'D: 595.71 | C: 13.2\n8× B300 SXM6 AC | 269 GB HBM3e\n64 CPU | 500 GB RAM | 1100 W TDP',
        },
    }

    _arch_cats    = ['B200', 'B300']
    _cluster_cats = ['Lenovo', 'Schmidt']
    _dtype_cats   = ['bf16', 'fp16', 'fp8']

    # All four CV models benchmarked
    CV_INF_MODELS = ['ResNet-50', 'EfficientNet-B4', 'ViT-Large', 'ConvNeXt-Base']
    CV_TRAIN_MODELS = ['ResNet-50', 'ViT-Large']
    MODEL_CLEAN_MAP = {
        'resnet50':       'ResNet-50',
        'efficientnet_b4':'EfficientNet-B4',
        'vit_l_16':       'ViT-Large',
        'convnext_base':  'ConvNeXt-Base',
    }

    # ── Data loading ──────────────────────────────────────────────────────────
    def load_and_tag(file, cluster):
        path = os.path.join('results', file)
        if os.path.exists(path):
            df = pd.read_csv(path).drop_duplicates()
            df['Cluster'] = cluster
            return df
        return pd.DataFrame()

    cv_df = pd.concat([
        load_and_tag('cv_benchmark_results_lenovo.csv',  'Lenovo'),
        load_and_tag('cv_benchmark_results_schmidt.csv', 'Schmidt'),
    ], ignore_index=True)

    cv_train_df = pd.concat([
        load_and_tag('cv_train_benchmark_results_lenovo.csv',  'Lenovo'),
        load_and_tag('cv_train_benchmark_results_schmidt.csv', 'Schmidt'),
    ], ignore_index=True)

    llm_df = pd.concat([
        load_and_tag('llm_benchmark_results_lenovo.csv',  'Lenovo'),
        load_and_tag('llm_benchmark_results_schmidt.csv', 'Schmidt'),
    ], ignore_index=True)

    if cv_df.empty or llm_df.empty:
        print("Error: Missing CSV files. Run from the Benchmarks/ directory.")
        return

    # ── Standardisation ───────────────────────────────────────────────────────
    def prep_cv(df, model_subset=None):
        df = df.copy()
        df['Arch'] = df['Tag'].apply(lambda x: str(x).split('_')[0].upper())
        df['GPUs'] = pd.to_numeric(df['GPUs'], errors='coerce').fillna(-1).astype(int)
        df = df[df['GPUs'] > 0]
        df['Model_Clean'] = df['Model'].replace(MODEL_CLEAN_MAP)
        df['Dtype'] = df['Dtype'].astype(str).str.lower()
        if model_subset:
            df = df[df['Model_Clean'].isin(model_subset)]
        df['Arch']    = pd.Categorical(df['Arch'],    categories=_arch_cats,    ordered=True)
        df['Cluster'] = pd.Categorical(df['Cluster'], categories=_cluster_cats, ordered=True)
        return df

    cv_df       = prep_cv(cv_df)
    cv_train_df = prep_cv(cv_train_df, CV_TRAIN_MODELS) if not cv_train_df.empty else cv_train_df

    llm_df['Arch'] = llm_df['Tag'].apply(lambda x: str(x).split('_')[0].upper())
    llm_df['GPUs'] = pd.to_numeric(llm_df['TP'], errors='coerce').fillna(-1).astype(int)
    llm_df = llm_df[llm_df['GPUs'] > 0]
    llm_df['Model_Clean'] = llm_df['Model'].apply(
        lambda x: '8B Model' if '8B' in str(x) else '70B Model')
    llm_df['Dtype'] = (llm_df['Dtype'].astype(str).str.lower()
                       .replace({'bfloat16': 'bf16', 'float16': 'fp16'}))
    llm_df['Arch']    = pd.Categorical(llm_df['Arch'],    categories=_arch_cats,    ordered=True)
    llm_df['Cluster'] = pd.Categorical(llm_df['Cluster'], categories=_cluster_cats, ordered=True)
    llm_df['Dtype']   = pd.Categorical(llm_df['Dtype'],   categories=_dtype_cats,   ordered=True)


    # ── Shared layout helpers ─────────────────────────────────────────────────
    def apply_specs_panel(g):
        spec_text = "HARDWARE SPECS\nD = Driver  C = CUDA\n" + "=" * 24 + "\n"
        for arch in _arch_cats:
            if arch in SPECS:
                spec_text += f"[{arch}]\n"
                for cluster, s in SPECS[arch].items():
                    spec_text += f" {cluster}:\n   {s.replace(chr(10), chr(10)+'   ')}\n"
                spec_text += "\n"
        g.figure.subplots_adjust(right=0.80, top=0.80, bottom=0.12, wspace=0.15)
        g.figure.text(0.82, 0.5, spec_text.strip(), fontsize=9, va='center', ha='left',
                      family='monospace',
                      bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da'))

    def safe_legend(g):
        if getattr(g, '_legend', None):
            g._legend.remove()
        for ax in g.axes.flat:
            if ax.get_legend():
                ax.get_legend().remove()
        handles = [
            Patch(facecolor=CLUSTER_COLORS['Lenovo'],  edgecolor='#1a1a1a', label='Lenovo'),
            Patch(facecolor=CLUSTER_COLORS['Schmidt'], edgecolor='#1a1a1a', label='Schmidt'),
        ]
        g.figure.legend(handles=handles, loc="lower center", ncol=2,
                        bbox_to_anchor=(0.41, 0.01), framealpha=0.95, handlelength=1.5)

    # ==========================================================================
    # FIG 1: CV Inference — Multi-GPU Peak Throughput (all 4 models, BF16)
    # ==========================================================================
    cv_inf = cv_df[cv_df['Model_Clean'].isin(CV_INF_MODELS)]
    cv_peak_bf16 = (cv_inf[cv_inf['Dtype'] == 'bf16']
                    .groupby(['Arch', 'Cluster', 'Model_Clean', 'GPUs'], observed=False)['FPS']
                    .max().reset_index())
    g1 = sns.catplot(data=cv_peak_bf16, x='Arch', y='FPS', hue='Cluster',
                     col='GPUs', row='Model_Clean', kind='bar',
                     palette=CLUSTER_COLORS, height=3.2, aspect=1.3, sharey='row',
                     row_order=CV_INF_MODELS)
    g1.set_axis_labels("", "Peak Throughput (FPS) ↑")
    g1.set_titles("{row_name} | {col_name} GPU(s)")
    g1.figure.suptitle("CV Inference — Multi-GPU Peak Throughput, BF16", y=0.98)
    apply_specs_panel(g1)
    safe_legend(g1)
    g1.savefig('figures/1_cv_inference_gpu_scaling_bf16.png', dpi=300,
               bbox_inches='tight')
    plt.close(g1.figure)

    # ==========================================================================
    # FIG 2: CV Inference — BF16 vs FP16 Peak Throughput (1 GPU, all models)
    # ==========================================================================
    cv_1g_prec = (cv_inf[cv_inf['GPUs'] == 1]
                  .groupby(['Arch', 'Cluster', 'Model_Clean', 'Dtype'], observed=False)['FPS']
                  .max().reset_index())
    # Use string columns so only present combos get bars
    for _col in ['Arch', 'Cluster', 'Dtype', 'Model_Clean']:
        cv_1g_prec[_col] = cv_1g_prec[_col].astype(str)
    g2 = sns.catplot(data=cv_1g_prec, x='Arch', y='FPS', hue='Cluster',
                     col='Dtype', row='Model_Clean', kind='bar',
                     palette=CLUSTER_COLORS, height=3.2, aspect=1.3, sharey='row',
                     order=_arch_cats, hue_order=_cluster_cats,
                     col_order=['bf16', 'fp16'], row_order=CV_INF_MODELS)
    g2.set_axis_labels("", "Peak Throughput (FPS) ↑")
    g2.set_titles("{row_name} | {col_name}")
    g2.figure.suptitle("CV Inference — BF16 vs FP16 (1 GPU, All Models)", y=0.98)
    apply_specs_panel(g2)
    safe_legend(g2)
    g2.savefig('figures/2_cv_inference_bf16_vs_fp16.png', dpi=300,
               bbox_inches='tight')
    plt.close(g2.figure)

    # ==========================================================================
    # FIG 3: CV Inference — Batch Saturation Curve (1 GPU, BF16, all models)
    # ==========================================================================
    cv_1g_line = cv_inf[(cv_inf['GPUs'] == 1) & (cv_inf['Dtype'] == 'bf16')
                        & (cv_inf['FPS'] > 0)]
    g3 = sns.relplot(data=cv_1g_line, x='Batch', y='FPS', hue='Cluster', col='Arch',
                     row='Model_Clean', kind='line', style='Cluster',
                     markers=True, dashes=False, markersize=7, linewidth=2.5,
                     palette=CLUSTER_COLORS, height=3.0, aspect=1.2,
                     facet_kws={'sharey': 'row'}, row_order=CV_INF_MODELS)
    g3.set(xscale="log")
    g3.set_axis_labels("Batch Size (log)", "Throughput (FPS) ↑")
    g3.set_titles("{row_name} | {col_name}")
    g3.figure.suptitle("CV Inference — Batch Saturation Curve (1 GPU, BF16)", y=0.98)
    apply_specs_panel(g3)
    safe_legend(g3)
    g3.savefig('figures/3_cv_inference_batch_saturation.png', dpi=300,
               bbox_inches='tight')
    plt.close(g3.figure)

    # ==========================================================================
    # FIG 4: CV Inference — Energy Efficiency (1 GPU, BF16, all models)
    # ==========================================================================
    g4 = sns.relplot(data=cv_1g_line, x='Batch', y='Img_per_Watt', hue='Cluster',
                     col='Arch', row='Model_Clean', kind='line', style='Cluster',
                     markers=True, dashes=False, markersize=7, linewidth=2.5,
                     palette=CLUSTER_COLORS, height=3.0, aspect=1.2,
                     facet_kws={'sharey': 'row'}, row_order=CV_INF_MODELS)
    g4.set(xscale="log")
    g4.set_axis_labels("Batch Size (log)", "Efficiency (Images / Watt) ↑")
    g4.set_titles("{row_name} | {col_name}")
    g4.figure.suptitle("CV Inference — Energy Efficiency (1 GPU, BF16)", y=0.98)
    apply_specs_panel(g4)
    safe_legend(g4)
    g4.savefig('figures/4_cv_inference_energy_efficiency.png', dpi=300,
               bbox_inches='tight')
    plt.close(g4.figure)

    # ==========================================================================
    # FIG 5: CV Training — Multi-GPU Peak Throughput (ResNet-50, ViT-Large)
    # ==========================================================================
    if not cv_train_df.empty:
        cv_train_peak = (cv_train_df
                         .groupby(['Arch', 'Cluster', 'Model_Clean', 'GPUs'], observed=False)['FPS']
                         .max().reset_index())
        g5 = sns.catplot(data=cv_train_peak, x='Arch', y='FPS', hue='Cluster',
                         col='GPUs', row='Model_Clean', kind='bar',
                         palette=CLUSTER_COLORS, height=4, aspect=1.3, sharey='row',
                         row_order=CV_TRAIN_MODELS)
        g5.set_axis_labels("", "Training Throughput (FPS) ↑")
        g5.set_titles("{row_name} | {col_name} GPU(s)")
        g5.figure.suptitle("CV Training — Multi-GPU Peak Throughput (DDP, BF16+FP16)", y=0.96)
        apply_specs_panel(g5)
        safe_legend(g5)
        g5.savefig('figures/5_cv_training_gpu_scaling.png', dpi=300,
                   bbox_inches='tight')
        plt.close(g5.figure)

    # ==========================================================================
    # FIG 6: LLM — Multi-GPU Capacity (BF16)
    # ==========================================================================
    llm_bf16 = (llm_df[llm_df['Dtype'] == 'bf16']
                .groupby(['Arch', 'Cluster', 'Model_Clean', 'GPUs'], observed=False)['TPS']
                .max().reset_index())
    g6 = sns.catplot(data=llm_bf16, x='Arch', y='TPS', hue='Cluster',
                     col='GPUs', row='Model_Clean', kind='bar',
                     palette=CLUSTER_COLORS, height=4, aspect=1.3, sharey='row')
    g6.set_axis_labels("", "Generation Speed (TPS) ↑")
    g6.set_titles("{row_name} | {col_name} GPU(s)")
    g6.figure.suptitle("LLM — Multi-GPU Capacity, BF16 Baseline", y=0.96)
    apply_specs_panel(g6)
    safe_legend(g6)
    g6.savefig('figures/6_llm_gpu_scaling.png', dpi=300, bbox_inches='tight')
    plt.close(g6.figure)

    # ==========================================================================
    # FIG 7: LLM — Precision Comparison BF16/FP16/FP8 (8B model)
    # ==========================================================================
    llm_prec = llm_df[llm_df['Model_Clean'] == '8B Model'].copy()
    for _col in ['Arch', 'Cluster', 'Dtype']:
        llm_prec[_col] = llm_prec[_col].astype(str)
    _present_dtypes = [d for d in _dtype_cats if d in llm_prec['Dtype'].values]
    g7 = sns.catplot(data=llm_prec, x='Arch', y='TPS', hue='Cluster',
                     col='Dtype', row='GPUs', kind='bar',
                     palette=CLUSTER_COLORS, height=3.5, aspect=1.2, sharey='row',
                     order=_arch_cats, hue_order=_cluster_cats,
                     col_order=_present_dtypes, row_order=[1, 4, 8])
    g7.set_axis_labels("", "Throughput (TPS) ↑")
    g7.set_titles("8B Model | {row_name} GPU(s) | {col_name}")
    g7.figure.suptitle("LLM — BF16 / FP16 / FP8 Precision Comparison (8B Model)", y=0.96)
    apply_specs_panel(g7)
    safe_legend(g7)
    g7.savefig('figures/7_llm_precision.png', dpi=300, bbox_inches='tight')
    plt.close(g7.figure)

    # ==========================================================================
    # FIG 8: LLM — Precision Comparison BF16/FP16/FP8 (70B model)
    # ==========================================================================
    llm_prec_70b = llm_df[llm_df['Model_Clean'] == '70B Model'].copy()
    for _col in ['Arch', 'Cluster', 'Dtype']:
        llm_prec_70b[_col] = llm_prec_70b[_col].astype(str)
    _present_dtypes_70b = [d for d in _dtype_cats if d in llm_prec_70b['Dtype'].values]
    g8 = sns.catplot(data=llm_prec_70b, x='Arch', y='TPS', hue='Cluster',
                     col='Dtype', row='GPUs', kind='bar',
                     palette=CLUSTER_COLORS, height=3.5, aspect=1.2, sharey='row',
                     order=_arch_cats, hue_order=_cluster_cats,
                     col_order=_present_dtypes_70b, row_order=[1, 4, 8])
    g8.set_axis_labels("", "Throughput (TPS) ↑")
    g8.set_titles("70B Model | {row_name} GPU(s) | {col_name}")
    g8.figure.suptitle("LLM — BF16 / FP16 / FP8 Precision Comparison (70B Model)", y=0.96)
    apply_specs_panel(g8)
    safe_legend(g8)
    g8.savefig('figures/8_llm_precision_70b.png', dpi=300, bbox_inches='tight')
    plt.close(g8.figure)

    # ==========================================================================
    # FIG 9: LLM — Time to First Token, BF16 (lower is better)
    # ==========================================================================
    llm_ttft = (llm_df[llm_df['Dtype'] == 'bf16']
                .groupby(['Arch', 'Cluster', 'Model_Clean', 'GPUs'], observed=False)['TTFT_ms']
                .min().reset_index())
    g9 = sns.catplot(data=llm_ttft, x='Arch', y='TTFT_ms', hue='Cluster',
                     col='GPUs', row='Model_Clean', kind='bar',
                     palette=CLUSTER_COLORS, height=4, aspect=1.3, sharey='row')
    g9.set_axis_labels("", "Time To First Token (ms) ↓")
    g9.set_titles("{row_name} | {col_name} GPU(s)")
    g9.figure.suptitle("LLM — Responsiveness: TTFT, BF16 (Lower is Better)", y=0.96)
    apply_specs_panel(g9)
    safe_legend(g9)
    g9.savefig('figures/9_llm_ttft.png', dpi=300, bbox_inches='tight')
    plt.close(g9.figure)

    # ==========================================================================
    # FIG 10: LLM — Speed vs. Latency Frontier (all dtypes)
    # ==========================================================================
    frontier = (llm_df[llm_df['TPS'] > 0]
                .groupby(['Arch', 'Cluster', 'Model_Clean', 'GPUs', 'Dtype'], observed=True)
                .agg({'TPS': 'max', 'Avg_Lat_ms': 'min'}).reset_index())
    g10 = sns.relplot(data=frontier, x='TPS', y='Avg_Lat_ms', hue='Cluster', style='Arch',
                      col='GPUs', row='Model_Clean', s=150, alpha=0.9,
                      palette=CLUSTER_COLORS, height=4.5, aspect=1.3,
                      facet_kws={'sharey': 'row', 'sharex': 'row'})
    g10.set_axis_labels("Throughput (TPS) ↑", "Average Latency (ms) ↓")
    g10.set_titles("{row_name} | {col_name} GPU(s)")
    g10.figure.suptitle("LLM — Inference Frontier: Throughput vs. Latency", y=0.96)
    apply_specs_panel(g10)
    safe_legend(g10)
    g10.savefig('figures/10_llm_frontier.png', dpi=300, bbox_inches='tight')
    plt.close(g10.figure)

    print("Done — 10 figures saved to figures/")
    print("  1  CV inference multi-GPU scaling (BF16, all 4 models)")
    print("  2  CV inference BF16 vs FP16 (1 GPU, all 4 models)")
    print("  3  CV inference batch saturation curve (1 GPU, BF16)")
    print("  4  CV inference energy efficiency (1 GPU, BF16)")
    print("  5  CV training multi-GPU scaling (DDP, ResNet-50 + ViT-Large)")
    print("  6  LLM multi-GPU capacity (BF16)")
    print("  7  LLM precision comparison BF16/FP16/FP8 (8B model)")
    print("  8  LLM precision comparison BF16/FP16/FP8 (70B model)")
    print("  9  LLM time-to-first-token (BF16)")
    print(" 10  LLM inference frontier: throughput vs latency")


if __name__ == "__main__":
    build_comprehensive_visuals()
