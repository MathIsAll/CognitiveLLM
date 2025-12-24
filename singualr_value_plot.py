# visualize_llama_ckpts_svd.py
import os

import math
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from transformers import AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
import scienceplots

plt.style.use(['science', 'nature', 'no-latex'])  # 'no-latex' 如果你不想使用 LaTeX 渲染字体


# ----------------------------
# 工具函数：读取指定层/模块的线性权重
# ----------------------------
def get_linear_weight(model, layer_idx: int, which: str) -> torch.Tensor:
    """
    返回指定层与模块的权重矩阵 (float32, CPU)
    which in {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
    """
    layer = model.model.layers[layer_idx]
    if which in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        lin = getattr(layer.self_attn, which)
    else:
        lin = getattr(layer.mlp, which)
    W = lin.weight.detach().float().cpu()  # [out, in]
    return W


def load_llama_model(ckpt_or_name: str):
    """
    加载LLaMA模型
    """
    cfg = AutoConfig.from_pretrained(ckpt_or_name)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_or_name,
        config=cfg,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None
    )
    model = model.cpu().eval()
    return model


# --------------------------------
# 4个核心度量函数
# --------------------------------
def singular_value_spectrum(W: torch.Tensor) -> torch.Tensor:
    """奇异值谱（降序）—— 度量#1：Singular Value Spectrum[1,2](@ref)"""
    s = torch.linalg.svdvals(W.to("cuda"))
    return torch.sort(s, descending=True).values.cpu()


def energy_ratio(W: torch.Tensor, k: int) -> float:
    """前k个奇异值能量占比E_k——度量#2：Energy Ratio[2](@ref)"""
    s = singular_value_spectrum(W.to("cuda"))
    k = min(k, s.numel())
    num = torch.sum(s[:k] ** 2)
    den = torch.sum(s ** 2) + 1e-12
    return (num / den).item()


def principal_angles_between_subspaces(W1: torch.Tensor, W2: torch.Tensor, k: int) -> torch.Tensor:
    """前k个左奇异向量子空间的主角集合（弧度，升序）——度量#3：Principal Angle[2](@ref)"""
    U1, _, _ = torch.linalg.svd(W1.to("cuda"))
    U2, _, _ = torch.linalg.svd(W2.to("cuda"))
    k = min(k, U1.shape[1], U2.shape[1])
    Q1, Q2 = U1[:, :k], U2[:, :k]
    M = Q1.T @ Q2
    cos_t = torch.linalg.svdvals(M).clamp(0.0, 1.0)
    thetas = torch.acos(cos_t)
    return torch.sort(thetas).values.detach().cpu().numpy()


def subspace_overlap(W1: torch.Tensor, W2: torch.Tensor, k: int) -> float:
    """Overlap = (1/k)||Q1^T Q2||_F^2 ∈ [0,1]——度量#4：Subspace Overlap[2](@ref)"""
    U1, _, _ = torch.linalg.svd(W1.to("cuda"))
    U2, _, _ = torch.linalg.svd(W2.to("cuda"))
    k = min(k, U1.shape[1], U2.shape[1])
    Q1, Q2 = U1[:, :k], U2[:, :k]
    M = Q1.T @ Q2
    frob2 = torch.sum(M ** 2)
    return (frob2 / (k + 1e-12)).item()


# --------------------------------
# 批量计算函数（返回字典格式）
# --------------------------------
def compute_metrics_series(
        models: List,
        layer_idx: int,
        which: str = "q_proj",
        k_subspace: int = 32,
        topn_singular: int = 8
) -> Tuple[Dict[str, np.ndarray], List[torch.Tensor]]:
    """
    计算时间序列度量（与baseline、与相邻对比）
    返回字典，其中每个键对应一个指标，值为NumPy数组[3](@ref)
    """
    assert len(models) >= 2, "需要至少两个checkpoints才能对比"

    weights = [get_linear_weight(m, layer_idx, which) for m in models]
    spectra = [singular_value_spectrum(W) for W in weights]
    max_topn = min(topn_singular, min(s.numel() for s in spectra))

    # 初始化存储列表
    t_list = []
    ek_list = []
    angle_mean_list = []
    angle_max_list = []
    overlap_list = []
    angle_mean_prev_list = []
    angle_max_prev_list = []
    overlap_prev_list = []
    sigma_lists = [[] for _ in range(max_topn)]

    W0 = weights[0]  # baseline权重
    for t in range(len(weights)):
        Wt = weights[t]

        # 计算与baseline的度量
        E_k = energy_ratio(Wt, k_subspace)
        angles = principal_angles_between_subspaces(W0, Wt, k_subspace)
        angles_deg = angles * 180.0 / math.pi
        overlap = subspace_overlap(W0, Wt, k_subspace)
        top_n_singular = singular_value_spectrum(Wt.to("cuda")).detach().cpu().numpy()[:topn_singular]

        # 收集数据
        t_list.append(top_n_singular)
        ek_list.append(E_k)
        angle_mean_list.append(float(angles_deg.mean()))
        angle_max_list.append(float(angles_deg.max()))
        overlap_list.append(overlap)

        # 与相邻ckpt的度量
        if t > 0:
            angles_adj = principal_angles_between_subspaces(weights[t - 1], Wt, k_subspace)
            angles_adj_deg = angles_adj * 180.0 / math.pi
            overlap_adj = subspace_overlap(weights[t - 1], Wt, k_subspace)
            angle_mean_prev_list.append(float(angles_adj_deg.mean()))
            angle_max_prev_list.append(float(angles_adj_deg.max()))
            overlap_prev_list.append(overlap_adj)
        else:
            # 第一个检查点没有前一个点，用NaN填充
            angle_mean_prev_list.append(np.nan)
            angle_max_prev_list.append(np.nan)
            overlap_prev_list.append(np.nan)

    # 构建返回的字典
    metrics_dict = {
        't': np.array(t_list).reshape(-1, 1, topn_singular),
        'E_k_vs_base': np.array(ek_list).reshape(len(models), -1),
        'angle_mean_vs_base_deg': np.array(angle_mean_list).reshape(len(models), -1),
        'angle_max_vs_base_deg': np.array(angle_max_list).reshape(len(models), -1),
        'overlap_vs_base': np.array(overlap_list).reshape(len(models), -1),
        # 'angle_mean_vs_prev_deg': np.array(angle_mean_prev_list),
        # 'angle_max_vs_prev_deg': np.array(angle_max_prev_list),
        # 'overlap_vs_prev': np.array(overlap_prev_list),
        'layer_idx': np.full(len(t_list), layer_idx)  # 添加层索引信息
    }

    return metrics_dict, spectra


# ----------------------------
# 绘图函数（使用字典数据）
# ----------------------------
def _num(x):
    """转换为数值类型"""
    return pd.to_numeric(x, errors="coerce")


def plot_top_singulars(all_metrics: Dict[str, np.ndarray], outdir: str, topn: int = 8, title_suffix: str = ""):
    """奇异值前top-n的演化（对数y轴）[1](@ref)"""
    num_subplots = all_metrics['t'].shape[0]
    with plt.style.context(['science', 'nature', 'no-latex']):
        fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 5))  # 1行2列，设置整个画布的大小

        for id, ax in enumerate(axs):
            im1 = ax.imshow(all_metrics['t'][id], cmap='viridis', aspect='auto')  # 'aspect='auto'' 让图像填满子图
            ax.set_title(f'Heatmap of Matrix {id}')
            ax.set_xlabel('Column Index')
            ax.set_ylabel('Row Index')
            fig.colorbar(im1, ax=ax, shrink=0.8)  # 为第一个子图添加颜色条

        # 自动调整子图布局，防止标签重叠
        plt.tight_layout()
        plt.title(f'Top-{topn} Singular Values over Checkpoints{title_suffix}')
        plt.legend()
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f'top{topn}_singulars.png'), dpi=160, bbox_inches='tight')
        plt.close()


def plot_energy_ratio(all_metrics: Dict[str, np.ndarray], outdir: str, title_suffix: str = ""):
    """能量占比E_k的演化"""
    with plt.style.context(['science', 'nature', 'no-latex']):
        groups = [f'Model-{id}' for id in range(all_metrics['E_k_vs_base'].shape[0])]  # 组的名称
        categories = [f'Layer-{id}' for id in range(all_metrics['E_k_vs_base'].shape[1])]  # 每组内各个柱子的名称

        # 2. 设置图形参数
        bar_width = 0.05  # 柱子的宽度
        index = np.arange(len(categories))  # 类别的索引，用于确定每组柱子的x轴基础位置

        # 3. 创建图形和坐标轴
        plt.figure(figsize=(8, 6))

        # 4. 绘制分组柱状图
        # 遍历每一组数据
        for i, group_data in enumerate(all_metrics['E_k_vs_base']):
            # 计算当前组每根柱子的x坐标：基础位置 + 该组的偏移量
            x_positions = index + i * bar_width
            # 绘制当前组的柱子
            plt.bar(x_positions, group_data, bar_width, label=groups[i])

        # 5. 添加图表标签和标题
        plt.xlabel('Categories')
        plt.ylabel('Values')
        plt.title(f'Energy Ratio vs Baseline over Checkpoints{title_suffix}')
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, 'energy_ratio.png'), dpi=160, bbox_inches='tight')
        plt.close()


def plot_angles_vs_base_multi(all_metrics: Dict[str, np.ndarray], outdir: str, title_suffix: str = ""):
    """所有层相对于baseline的主角统计图（使用字典数据）"""
    # 获取唯一的层索引
    layer_indices = np.unique(all_metrics['layer_idx'])

    # 绘制均值角度
    with plt.style.context(['science', 'nature', 'no-latex']):
        groups = [f'Model-{id}' for id in range(all_metrics['angle_mean_vs_base_deg'].shape[0])]  # 组的名称
        categories = [f'Layer-{id}' for id in range(all_metrics['angle_mean_vs_base_deg'].shape[1])]  # 每组内各个柱子的名称

        # 2. 设置图形参数
        bar_width = 0.05  # 柱子的宽度
        index = np.arange(len(categories))  # 类别的索引，用于确定每组柱子的x轴基础位置

        # 3. 创建图形和坐标轴
        plt.figure(figsize=(8, 6))

        # 4. 绘制分组柱状图
        # 遍历每一组数据
        for i, group_data in enumerate(all_metrics['angle_mean_vs_base_deg']):
            # 计算当前组每根柱子的x坐标：基础位置 + 该组的偏移量
            x_positions = index + i * bar_width
            # 绘制当前组的柱子
            plt.bar(x_positions, group_data, bar_width, label=groups[i])
        plt.xlabel('checkpoint index (t)')
        plt.ylabel('mean angle vs base (deg)')
        plt.title(f'Principal Angles (mean) vs Baseline{title_suffix}')
        plt.legend(ncol=2, fontsize=8)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, 'angles_vs_base_mean_all_layers.png'), dpi=160, bbox_inches='tight')
        plt.close()

    # 绘制最大角度
    with plt.style.context(['science', 'nature', 'no-latex']):
        groups = [f'Model-{id}' for id in range(all_metrics['angle_max_vs_base_deg'].shape[0])]  # 组的名称
        categories = [f'Layer-{id}' for id in range(all_metrics['angle_max_vs_base_deg'].shape[1])]  # 每组内各个柱子的名称

        # 2. 设置图形参数
        bar_width = 0.05  # 柱子的宽度
        index = np.arange(len(categories))  # 类别的索引，用于确定每组柱子的x轴基础位置

        # 3. 创建图形和坐标轴
        plt.figure(figsize=(8, 6))

        # 4. 绘制分组柱状图
        # 遍历每一组数据
        for i, group_data in enumerate(all_metrics['angle_max_vs_base_deg']):
            # 计算当前组每根柱子的x坐标：基础位置 + 该组的偏移量
            x_positions = index + i * bar_width
            # 绘制当前组的柱子
            plt.bar(x_positions, group_data, bar_width, label=groups[i])
        plt.xlabel('checkpoint index (t)')
        plt.ylabel('max angle vs base (deg)')
        plt.title(f'Principal Angles (max) vs Baseline{title_suffix}')
        plt.legend(ncol=2, fontsize=8)
        plt.savefig(os.path.join(outdir, 'angles_vs_base_max_all_layers.png'), dpi=160, bbox_inches='tight')
        plt.close()


def plot_angles_vs_prev_multi(all_metrics: Dict[str, np.ndarray], outdir: str, title_suffix: str = ""):
    """所有层相对于前一个检查点的主角统计图"""
    layer_indices = np.unique(all_metrics['layer_idx'])

    # 绘制均值角度
    with plt.style.context(['science', 'nature', 'no-latex']):
        plt.figure(figsize=(12, 8))
        for layer_idx in layer_indices:
            layer_mask = all_metrics['layer_idx'] == layer_idx
            x = all_metrics['t'][layer_mask]
            y = all_metrics['angle_mean_vs_prev_deg'][layer_mask]
            # 过滤掉NaN值（第一个检查点）
            valid_mask = ~np.isnan(y)
            if np.any(valid_mask):
                plt.plot(x[valid_mask], y[valid_mask], marker='o', label=f'layer{layer_idx}')

        plt.xlabel('checkpoint index (t)')
        plt.ylabel('mean angle vs prev (deg)')
        plt.title(f'Principal Angles (mean) vs Previous{title_suffix}')
        plt.legend(ncol=2, fontsize=8)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, 'angles_vs_prev_mean_all_layers.png'), dpi=160, bbox_inches='tight')
        plt.close()

    # 绘制最大角度
    with plt.style.context(['science', 'nature', 'no-latex']):
        plt.figure(figsize=(12, 8))
        for layer_idx in layer_indices:
            layer_mask = all_metrics['layer_idx'] == layer_idx
            x = all_metrics['t'][layer_mask]
            y = all_metrics['angle_max_vs_prev_deg'][layer_mask]
            valid_mask = ~np.isnan(y)
            if np.any(valid_mask):
                plt.plot(x[valid_mask], y[valid_mask], marker='o', label=f'layer{layer_idx}')

        plt.xlabel('checkpoint index (t)')
        plt.ylabel('max angle vs prev (deg)')
        plt.title(f'Principal Angles (max) vs Previous{title_suffix}')
        plt.legend(ncol=2, fontsize=8)
        plt.savefig(os.path.join(outdir, 'angles_vs_prev_max_all_layers.png'), dpi=160, bbox_inches='tight')
        plt.close()


def plot_overlap_vs_base_multi(all_metrics: Dict[str, np.ndarray], outdir: str, title_suffix: str = ""):
    """所有层相对于baseline的重叠度统计图"""
    layer_indices = np.unique(all_metrics['layer_idx'])

    with plt.style.context(['science', 'nature', 'no-latex']):
        groups = [f'Model-{id}' for id in range(all_metrics['overlap_vs_base'].shape[0])]  # 组的名称
        categories = [f'Layer-{id}' for id in range(all_metrics['overlap_vs_base'].shape[1])]  # 每组内各个柱子的名称

        # 2. 设置图形参数
        bar_width = 0.05  # 柱子的宽度
        index = np.arange(len(categories))  # 类别的索引，用于确定每组柱子的x轴基础位置

        # 3. 创建图形和坐标轴
        plt.figure(figsize=(8, 6))

        # 4. 绘制分组柱状图
        # 遍历每一组数据
        for i, group_data in enumerate(all_metrics['overlap_vs_base']):
            # 计算当前组每根柱子的x坐标：基础位置 + 该组的偏移量
            x_positions = index + i * bar_width
            # 绘制当前组的柱子
            plt.bar(x_positions, group_data, bar_width, label=groups[i])
        plt.xlabel('checkpoint index (t)')
        plt.ylabel('overlap vs base [0,1]')
        plt.title(f'Subspace Overlap vs Baseline{title_suffix}')
        plt.legend(ncol=2, fontsize=8)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, 'overlap_vs_base_all_layers.png'), dpi=160, bbox_inches='tight')
        plt.close()


def plot_overlap_vs_prev_multi(all_metrics: Dict[str, np.ndarray], outdir: str, title_suffix: str = ""):
    """所有层相对于前一个检查点的重叠度统计图"""
    layer_indices = np.unique(all_metrics['layer_idx'])

    with plt.style.context(['science', 'nature', 'no-latex']):
        plt.figure(figsize=(12, 8))
        for layer_idx in layer_indices:
            layer_mask = all_metrics['layer_idx'] == layer_idx
            x = all_metrics['t'][layer_mask]
            y = all_metrics['overlap_vs_prev'][layer_mask]
            valid_mask = ~np.isnan(y)
            if np.any(valid_mask):
                plt.plot(x[valid_mask], y[valid_mask], marker='o', label=f'layer{layer_idx}')

        plt.xlabel('checkpoint index (t)')
        plt.ylabel('overlap vs prev [0,1]')
        plt.title(f'Subspace Overlap vs Previous{title_suffix}')
        plt.legend(ncol=2, fontsize=8)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, 'overlap_vs_prev_all_layers.png'), dpi=160, bbox_inches='tight')
        plt.close()


# ----------------------------
# 主流程
# ----------------------------
if __name__ == "__main__":
    # 配置检查点路径
    checkpoints = [
        "/data_net/models_for_all/Qwen2.5-0.5B/",
        "/data_net/models_for_all/Qwen2.5-0.5B/",
        "/data_net/models_for_all/Qwen2.5-0.5B/",
        "/data_net/models_for_all/Qwen2.5-0.5B/",
    ]

    which = "q_proj"  # 分析哪个投影矩阵
    k_subspace = 8  # 子空间维度
    topn_singular = 8  # 保留的前n个奇异值

    # 加载所有模型
    print("Loading models...")
    models = [load_llama_model(p) for p in checkpoints]
    num_layers = len(models[0].model.layers)

    # 初始化一个空字典来收集所有层的指标
    all_metrics = {}

    # 处理每一层（这里示例只处理前5层）
    for layer_idx in tqdm(range(num_layers)):
        print(f"Processing layer {layer_idx} ...")
        metrics_dict, spectra = compute_metrics_series(
            models,
            layer_idx=layer_idx,
            which=which,
            k_subspace=k_subspace,
            topn_singular=topn_singular
        )

        # 如果是第一层，初始化所有数组
        if not all_metrics:
            for key in metrics_dict.keys():
                all_metrics[key] = metrics_dict[key]
        else:
            # 后续层，拼接数组
            for key in metrics_dict.keys():
                if key == 't':
                    all_metrics[key] = np.concatenate((all_metrics[key], metrics_dict[key]), axis=1)
                else:
                    all_metrics[key] = np.concatenate((all_metrics[key], metrics_dict[key]), axis=-1)

    # 创建输出目录
    outdir = f"svd_vis_all_layers_{which}"
    os.makedirs(outdir, exist_ok=True)

    # 保存为NPZ文件（可选）
    np.savez(os.path.join(outdir, 'metrics_all_layers.npz'), **all_metrics)

    # 绘制所有图表
    suffix = f" ({which}, k={k_subspace})"

    # 基础图表
    plot_top_singulars(all_metrics, outdir, topn=topn_singular, title_suffix=suffix)
    plot_energy_ratio(all_metrics, outdir, title_suffix=suffix)

    # 多层对比图表
    plot_angles_vs_base_multi(all_metrics, outdir, title_suffix=suffix)
    # plot_angles_vs_prev_multi(all_metrics, outdir, title_suffix=suffix)
    plot_overlap_vs_base_multi(all_metrics, outdir, title_suffix=suffix)
    # plot_overlap_vs_prev_multi(all_metrics, outdir, title_suffix=suffix)

    print(f"Done. Results saved under: {outdir}")
