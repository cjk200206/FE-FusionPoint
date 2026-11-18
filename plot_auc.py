import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

# ==========================================
# ⚙️ 排版模式切换开关
# 'double' = 双栏跨栏大图 (2行3列，适合 \begin{figure*})
# 'single' = 单栏半栏紧凑图 (3行2列，适合 \begin{figure})
# ==========================================
LAYOUT_MODE = 'single'  # <--- 在这里切换 'single' 或 'double'

# 1. 数据准备 (保持不变)
sequence_names = ["boxes_6dof", "poster_6dof", "dynamic_6dof", "hdr_boxes", "hdr_poster", "All (Average)"]
blur_thresholds = [1, 3, 5, 7] 

data_dict = {
    'SuperPoint': [
        [49.76, 34.56, 27.80, 24.13], 
        [74.10, 61.24, 50.52, 41.56], 
        [49.20, 35.37, 30.85, 29.09], 
        [43.94, 33.89, 24.72, 15.80], 
        [63.36, 57.97, 54.83, 55.40],
        [56.07, 44.61, 37.74, 33.20]
    ],
    'SuperEvent': [
        [39.10, 24.73, 19.14, 15.22], 
        [61.81, 47.27, 36.91, 32.67], 
        [48.84, 34.62, 30.93, 28.97], 
        [42.54, 27.90, 21.24, 16.35], 
        [62.58, 56.27, 56.25, 54.19],
        [50.98, 38.16, 32.89, 29.48]
    ],
    'FE-DeTr': [
        [31.72, 25.70, 17.42, 11.39], 
        [60.09, 39.19, 26.21, 18.95], 
        [38.97, 22.11, 16.36, 11.73], 
        [37.92, 22.80, 17.96, 13.71], 
        [49.57, 36.68, 30.43, 24.46],
        [43.65, 29.30, 21.68, 16.05]
    ],
    'Robust-DeTr': [
        [36.33, 23.25, 17.85, 15.57],
        [62.48, 47.71, 37.14, 32.61],
        [43.25, 29.53, 24.03, 22.49],
        [41.09, 28.05, 21.15, 13.37],
        [63.04, 55.72, 50.97, 49.31],
        [49.24, 36.85, 30.23, 26.67]
    ],
    'ALIKED': [
        [53.35, 39.84, 32.06, 26.89], 
        [76.38, 63.61, 54.40, 44.05], 
        [51.52, 35.62, 31.10, 30.45], 
        [46.05, 33.58, 27.18, 20.03], 
        [63.45, 61.14, 57.66, 55.67],
        [58.15, 46.76, 40.48, 35.42]
    ],
    'Ours': [
        [51.72, 38.99, 33.04, 30.05], 
        [75.39, 63.02, 54.17, 43.82], 
        [52.46, 41.72, 38.04, 35.96], 
        [49.13, 36.14, 29.44, 20.56], 
        [74.74, 68.65, 64.30, 61.91],
        [60.69, 49.70, 43.80, 38.46]
    ]
}

# 2. 根据模式自动配置排版参数
if LAYOUT_MODE == 'double':
    n_rows, n_cols = 2, 3
    fig_size = (15, 6)
    base_fontsize = 12
    title_fontsize = 13
    legend_cols = 5
    bbox_y = 1.08        # 图例的高度微调
    marker_sz = 6
    w_space, h_space = 0.06, 0.25
elif LAYOUT_MODE == 'single':
    n_rows, n_cols = 2, 3
    # 宽度设为 3.8 英寸以适应单栏，高度适当拉长
    fig_size = (7.0, 4.5) 
    base_fontsize = 8    # 单栏必须缩小字号，否则会重叠
    title_fontsize = 10
    legend_cols = 6      # 单栏太窄，图例分成两列显示
    bbox_y = 1.00
    marker_sz = 3        # 缩小点的大小
    w_space, h_space = 0.1, 0.2
else:
    raise ValueError("LAYOUT_MODE 必须是 'single' 或 'double'")

plt.rcParams.update({'font.size': base_fontsize, 'font.family': 'serif'})
fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharey=True, sharex=True)
# fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharey=False, sharex=True)
axes_flat = axes.flatten()

styles = {
    'SuperPoint': {'color': '#1f77b4', 'ls': '--', 'marker': 's', 'lw': 1.2},
    'SuperEvent': {'color': '#2ca02c', 'ls': '-.', 'marker': '^', 'lw': 1.2},
    'FE-DeTr':    {'color': '#7f7f7f', 'ls': ':',  'marker': 'x', 'lw': 1.2},
    'Robust-DeTr':{'color': '#9467bd', 'ls': ':',  'marker': '*', 'lw': 1.2},
    'ALIKED':     {'color': "#ffaf0e", 'ls': '--', 'marker': 'd', 'lw': 1.2},
    'Ours':       {'color': '#d62728', 'ls': '-',  'marker': 'o', 'lw': 2.0}
}

# 3. 遍历画图 (逻辑自动适配单/双栏)
for i, seq_name in enumerate(sequence_names):
    ax = axes_flat[i]
    for method_name, method_data in data_dict.items():
        ax.plot(blur_thresholds, method_data[i], label=method_name, 
                color=styles[method_name]['color'], 
                linestyle=styles[method_name]['ls'], 
                linewidth=styles[method_name]['lw'], 
                marker=styles[method_name]['marker'],
                markersize=marker_sz if method_name != 'Ours' else marker_sz + 1.5)
    
    ax.set_xticks(blur_thresholds)
    
    # 动态判断是否在最后一行，如果在则显示 X 轴标签
    if i >= len(sequence_names) - n_cols:
        ax.set_xlabel(r'Blur Thr. $\theta^\circ$') # 缩写 Thr. 节省单栏空间
    
    clean_title = seq_name.replace('_', ' ').title()
    if i == 5:
        ax.set_title(clean_title, fontsize=title_fontsize, fontweight='bold', color='#d62728')
    else:
        ax.set_title(clean_title, fontsize=title_fontsize)
        
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 动态判断是否在最左侧一列，如果在则显示 Y 轴标签
    if i % n_cols == 0:
        ax.set_ylabel('AUC@10 (%)', fontweight='bold')

plt.subplots_adjust(wspace=w_space, hspace=h_space)
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=legend_cols, bbox_to_anchor=(0.5, bbox_y), frameon=False)

# 保存文件 (文件名带上模式方便区分)
filename = f'auc_sequence_{LAYOUT_MODE}.pdf'
plt.savefig(filename, format='pdf', bbox_inches='tight')
print(f"图像已成功保存为 {filename}")