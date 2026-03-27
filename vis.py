import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def build_comprehensive_visuals():
    if not os.path.exists('figs_comp'): os.makedirs('figs_comp') 
    
    # --- CORPORATE AESTHETICS ---
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({'figure.autolayout': True, 'axes.titleweight': 'bold', 'axes.titlesize': 18})

    ARCH_COLORS = {'A100': '#7F7F7F', 'H100': '#1F77B4', 'B200': '#FF7F0E', 'B300': '#2CA02C'}
    DTYPE_COLORS = {'bf16': '#4C72B0', 'fp16': '#55A868', 'fp8': '#C44E52', 'fp4': '#8172B3'}

    # --- DATA LOADING & STANDARDIZATION ---
    cv_df = pd.read_csv('cv_benchmark_results_0.csv').drop_duplicates()
    llm_df = pd.read_csv('llm_benchmark_results_0.csv').drop_duplicates()

    cv_df['Arch'] = cv_df['Tag'].apply(lambda x: str(x).split('_')[0].upper())
    cv_df['GPUs'] = pd.to_numeric(cv_df['GPUs'], errors='coerce')
    cv_df['Model_Clean'] = cv_df['Model'].replace({'resnet50': 'ResNet-50', 'vit_l_16': 'ViT-Large'})
    cv_df['Dtype'] = cv_df['Dtype'].astype(str).str.lower()
    
    llm_df['Arch'] = llm_df['Tag'].apply(lambda x: str(x).split('_')[0].upper())
    llm_df['GPUs'] = pd.to_numeric(llm_df['TP'], errors='coerce')
    llm_df['Model_Clean'] = llm_df['Model'].apply(lambda x: '8B Model' if '8B' in str(x) else '70B Model')
    llm_df['Dtype'] = llm_df['Dtype'].astype(str).str.lower().replace({'bfloat16': 'bf16', 'float16': 'fp16'})

    arch_order = ['A100', 'H100', 'B200', 'B300']
    cv_df['Arch'] = pd.Categorical(cv_df['Arch'], categories=[a for a in arch_order if a in cv_df['Arch'].unique()], ordered=True)
    llm_df['Arch'] = pd.Categorical(llm_df['Arch'], categories=[a for a in arch_order if a in llm_df['Arch'].unique()], ordered=True)

    def add_labels(ax, fmt="{:,.0f}", fontsize=11):
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(fmt.format(p.get_height()), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', fontsize=fontsize, fontweight='bold', 
                            color='#333333', xytext=(0, 4), textcoords='offset points')

    # FIG 1: CV Peak Throughput (GPU Scaling)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    cv_peak = cv_df.groupby(['Arch', 'Model_Clean', 'GPUs'], observed=True)['FPS'].max().reset_index()
    for i, m in enumerate(['ResNet-50', 'ViT-Large']):
        ax = sns.barplot(data=cv_peak[cv_peak['Model_Clean']==m], x='GPUs', y='FPS', hue='Arch', palette=ARCH_COLORS, ax=axes[i])
        add_labels(ax)
        ax.set_title(f'{m}: Throughput Scaling (FPS \u2191)')
        ax.set_xlabel('Number of GPUs')
        if i>0: ax.get_legend().remove()
    plt.suptitle('Computer Vision: Multi-GPU Processing Capacity', fontsize=22, fontweight='bold')
    plt.savefig('figs_comp/1_cv_gpu_scaling.png', bbox_inches='tight', dpi=300)
    plt.close()

    # FIG 2: CV Batch Saturation (Strict Pre-aggregation applied)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    cv_1g = cv_df[(cv_df['GPUs'] == 1) & (cv_df['Dtype'] == 'bf16')]
    cv_1g_agg = cv_1g.groupby(['Arch', 'Model_Clean', 'Batch'], observed=True)['FPS'].max().reset_index()
    for i, m in enumerate(['ResNet-50', 'ViT-Large']):
        ax = sns.lineplot(data=cv_1g_agg[cv_1g_agg['Model_Clean']==m], x='Batch', y='FPS', hue='Arch', 
                          style='Arch', markers=True, markersize=14, linewidth=3.5, palette=ARCH_COLORS, ax=axes[i])
        ax.set_xscale('log', base=2)
        ax.set_title(f'{m}: Saturation Curve (FPS \u2191)')
        ax.set_xlabel('Batch Size (Log Scale \u2192)')
        if i>0: ax.get_legend().remove()
    plt.suptitle('Hardware Physics: Batch Size vs. Throughput (1 GPU, BF16)', fontsize=22, fontweight='bold')
    plt.savefig('figs_comp/2_cv_batch_saturation.png', bbox_inches='tight', dpi=300)
    plt.close()

    # FIG 3: CV Energy Efficiency (Strict Pre-aggregation applied)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    cv_eff_agg = cv_1g.groupby(['Arch', 'Model_Clean', 'Batch'], observed=True)['Img_per_Watt'].max().reset_index()
    for i, m in enumerate(['ResNet-50', 'ViT-Large']):
        ax = sns.lineplot(data=cv_eff_agg[cv_eff_agg['Model_Clean']==m], x='Batch', y='Img_per_Watt', hue='Arch', 
                          style='Arch', markers=True, markersize=14, linewidth=3.5, palette=ARCH_COLORS, ax=axes[i])
        ax.set_xscale('log', base=2)
        ax.set_title(f'{m}: Efficiency vs. Load (Img/Watt \u2191)')
        ax.set_xlabel('Batch Size (Log Scale \u2192)')
        if i>0: ax.get_legend().remove()
    plt.suptitle('Sustainability: Energy Efficiency Curve (1 GPU, BF16)', fontsize=22, fontweight='bold')
    plt.savefig('figs_comp/3_cv_efficiency_curve.png', bbox_inches='tight', dpi=300)
    plt.close()

    # FIG 4: LLM Peak Generation Speed (BF16 Baseline)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    llm_bf16 = llm_df[llm_df['Dtype'] == 'bf16'].groupby(['Arch', 'Model_Clean', 'GPUs'], observed=True)['TPS'].max().reset_index()
    for i, m in enumerate(['8B Model', '70B Model']):
        ax = sns.barplot(data=llm_bf16[llm_bf16['Model_Clean']==m], x='GPUs', y='TPS', hue='Arch', palette=ARCH_COLORS, ax=axes[i])
        add_labels(ax)
        ax.set_title(f'{m}: Generation Speed (TPS \u2191)')
        ax.set_xlabel('Tensor Parallelism (GPUs)')
        if i>0: ax.get_legend().remove()
    plt.suptitle('Large Language Models: Multi-GPU Capacity (BF16 Baseline)', fontsize=22, fontweight='bold')
    plt.savefig('figs_comp/4_llm_gpu_scaling.png', bbox_inches='tight', dpi=300)
    plt.close()

    # FIG 5: LLM Precision Acceleration
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    llm_prec = llm_df[llm_df['Model_Clean'] == '8B Model']
    for i, tp in enumerate([1, 4]):
        subset = llm_prec[llm_prec['GPUs']==tp].groupby(['Arch', 'Dtype'], observed=True)['TPS'].max().reset_index()
        ax = sns.barplot(data=subset, x='Arch', y='TPS', hue='Dtype', 
                         hue_order=['bf16', 'fp16', 'fp8', 'fp4'], palette=DTYPE_COLORS, ax=axes[i])
        add_labels(ax)
        ax.set_title(f'8B Model Scale: {tp} GPU(s)')
        ax.set_xlabel('Architecture')
        if i>0: ax.get_legend().remove()
    plt.suptitle('Unlocking the Tensor Engine: The FP8/FP4 Multiplier (TPS \u2191)', fontsize=22, fontweight='bold')
    plt.savefig('figs_comp/5_llm_precision.png', bbox_inches='tight', dpi=300)
    plt.close()

    # FIG 6: LLM Real-Time Responsiveness
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    llm_ttft = llm_df[llm_df['Dtype'] == 'bf16'].groupby(['Arch', 'Model_Clean', 'GPUs'], observed=True)['TTFT_ms'].min().reset_index()
    for i, m in enumerate(['8B Model', '70B Model']):
        ax = sns.barplot(data=llm_ttft[llm_ttft['Model_Clean']==m], x='GPUs', y='TTFT_ms', hue='Arch', palette=ARCH_COLORS, ax=axes[i])
        add_labels(ax, fmt="{:,.1f}ms")
        ax.set_title(f'{m}: Latency Delay (ms \u2193)')
        ax.set_xlabel('Tensor Parallelism (GPUs)')
        if i>0: ax.get_legend().remove()
    plt.suptitle('Responsiveness: Time to First Token (TTFT, Lower is Better)', fontsize=22, fontweight='bold')
    plt.savefig('figs_comp/6_llm_ttft.png', bbox_inches='tight', dpi=300)
    plt.close()

    # FIG 7: The "Efficiency Frontier" (Strict Dtype grouping applied)
    plt.figure(figsize=(14, 7))
    frontier = llm_df.groupby(['Arch', 'Model_Clean', 'GPUs', 'Dtype'], observed=True).agg({'TPS': 'max', 'Avg_Lat_ms': 'min'}).reset_index()
    
    ax = sns.scatterplot(data=frontier, x='TPS', y='Avg_Lat_ms', hue='Arch', style='Model_Clean', 
                         size='GPUs', sizes=(150, 650), alpha=0.9, edgecolor='black', linewidth=1.5, palette=ARCH_COLORS)
    
    plt.annotate('The Productivity Zone\n(High Throughput, Low Latency)', 
                 xy=(frontier['TPS'].max()*0.85, frontier['Avg_Lat_ms'].min()*1.5), 
                 xytext=(frontier['TPS'].max()*0.5, frontier['Avg_Lat_ms'].max()*0.4),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10),
                 fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#F1C40F', alpha=0.3))
                 
    plt.title('LLM Inference Frontier: Generational Speed vs. Real-Time Latency', fontweight='bold', pad=15)
    plt.xlabel('Throughput (TPS) [\u2192 Higher is Better]')
    plt.ylabel('[\u2190 Lower is Better] Average Latency (ms)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('figs_comp/7_llm_frontier.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("Success: Visuals Generated.")

if __name__ == "__main__":
    build_comprehensive_visuals()