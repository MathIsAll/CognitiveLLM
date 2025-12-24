# visualize_llama_ckpts_svd.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from transformers import AutoConfig, AutoModelForCausalLM
from tqdm import tqdm


# ----------------------------
# 工具：读取指定层/模块的线性权重
# ----------------------------
def get_linear_weight(model, layer_idx: int, which: str) -> torch.Tensor:
    """
    返回指定层与模块的权重矩阵 (float32, CPU):
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
# 4 个度量
# --------------------------------
def singular_value_spectrum(W: torch.Tensor) -> torch.Tensor:
    """奇异值谱（降序）。—— 度量#1：Singular Value Spectrum"""
    s = torch.linalg.svdvals(W.to("cuda:0"))
    return torch.sort(s, descending=True).values.cpu()


def energy_ratio(W: torch.Tensor, k: int) -> float:
    """前 k 个奇异值能量占比 E_k。—— 度量#2：Energy Ratio"""
    s = singular_value_spectrum(W.to("cuda:0"))
    k = min(k, s.numel())
    num = torch.sum(s[:k] ** 2)
    den = torch.sum(s ** 2) + 1e-12
    return (num / den).item()


def principal_angles_between_subspaces(W1: torch.Tensor, W2: torch.Tensor, k: int) -> torch.Tensor:
    """前 k 个左奇异向量子空间的主角集合（弧度，升序）。—— 度量#3：Principal Angle"""
    U1, _, _ = torch.linalg.svd(W1.to("cuda:0"))
    U2, _, _ = torch.linalg.svd(W2.to("cuda:0"))
    k = min(k, U1.shape[1], U2.shape[1])
    Q1, Q2 = U1[:, :k], U2[:, :k]
    M = Q1.T @ Q2
    cos_t = torch.linalg.svdvals(M).clamp(0.0, 1.0)
    thetas = torch.acos(cos_t)
    return torch.sort(thetas).values.detach().cpu().numpy()


def subspace_overlap(W1: torch.Tensor, W2: torch.Tensor, k: int) -> float:
    """Overlap = (1/k)||Q1^T Q2||_F^2 ∈ [0,1]。—— 度量#4：Subspace Overlap"""
    U1, _, _ = torch.linalg.svd(W1.to("cuda:0"))
    U2, _, _ = torch.linalg.svd(W2.to("cuda:0"))
    k = min(k, U1.shape[1], U2.shape[1])
    Q1, Q2 = U1[:, :k], U2[:, :k]
    M = Q1.T @ Q2
    frob2 = torch.sum(M ** 2)
    return (frob2 / (k + 1e-12)).item()


# --------------------------------
# 批量计算：与 baseline、与相邻 ckpt 的对比
# --------------------------------
def compute_metrics_series(
        models: List,
        layer_idx: int,
        which: str = "q_proj",
        k_subspace: int = 32,
        topn_singular: int = 8
) -> Tuple[pd.DataFrame, List[torch.Tensor]]:
    """
    返回：
      - df：按 checkpoint 索引的时间序列度量（与 baseline、与相邻对比）
      - spectra_list：每个 ckpt 的奇异值谱（Tensor）
    """
    assert len(models) >= 2, "需要至少两个 checkpoints 才能对比"

    weights = [get_linear_weight(m, layer_idx, which) for m in models]

    # 预计算每个 ckpt 的奇异值谱
    spectra = [singular_value_spectrum(W) for W in weights]
    max_topn = min(topn_singular, min(s.numel() for s in spectra))

    rows = []
    # 基准
    W0 = weights[0]
    for t in range(len(weights)):
        Wt = weights[t]
        # 与 baseline 的度量
        E_k = energy_ratio(Wt, k_subspace)
        angles = principal_angles_between_subspaces(W0, Wt, k_subspace)
        angles_deg = angles * 180.0 / math.pi
        overlap = subspace_overlap(W0, Wt, k_subspace)

        row = {
            "t": t,
            # 如果你需要 ckpt 标签，这里可以外部传入；为了兼容原逻辑，先留空或使用 t
            "ckpt": t,
            "E_k_vs_base": E_k,  # 能量占比（绝对，非“相对”）
            "angle_mean_vs_base_deg": float(angles_deg.mean()),
            "angle_max_vs_base_deg": float(angles_deg.max()),
            "overlap_vs_base": overlap,
        }

        # 与相邻 ckpt 的度量
        if t > 0:
            angles_adj = principal_angles_between_subspaces(weights[t - 1], Wt, k_subspace)
            angles_adj_deg = angles_adj * 180.0 / math.pi
            overlap_adj = subspace_overlap(weights[t - 1], Wt, k_subspace)
            row.update({
                "angle_mean_vs_prev_deg": float(angles_adj_deg.mean()),
                "angle_max_vs_prev_deg": float(angles_adj_deg.max()),
                "overlap_vs_prev": overlap_adj,
            })
        else:
            row.update({
                "angle_mean_vs_prev_deg": None,
                "angle_max_vs_prev_deg": None,
                "overlap_vs_prev": None,
            })

        # 前 top-n 奇异值（便于可视化演化）
        s = spectra[t]
        for i in range(max_topn):
            row[f"sigma{i + 1}"] = s[i].item()

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, spectra


# ----------------------------
# 单层绘图（保留）
# ----------------------------
def plot_top_singulars(df: pd.DataFrame, outdir: str, topn: int = 8, title_suffix: str = ""):
    """奇异值前 top-n 的演化（对数 y 轴）"""
    plt.figure()
    for i in range(1, topn + 1):
        col = f"sigma{i}"
        if col in df.columns:
            plt.plot(df["t"], df[col], marker="o", label=col)
    plt.yscale("log")
    plt.xlabel("checkpoint index (t)")
    plt.ylabel("top singular values")
    plt.title(f"Top-{topn} Singular Values over Checkpoints{title_suffix}")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"top{topn}_singulars.png"), dpi=160, bbox_inches="tight")
    plt.close()


def plot_energy_ratio(df: pd.DataFrame, outdir: str, title_suffix: str = ""):
    """能量占比 E_k 的演化"""
    plt.figure()
    plt.plot(df["t"], df["E_k_vs_base"], marker="o")
    plt.xlabel("checkpoint index (t)")
    plt.ylabel("Energy Ratio E_k")
    plt.title(f"Energy Ratio vs Baseline over Checkpoints{title_suffix}")
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "energy_ratio.png"), dpi=160, bbox_inches="tight")
    plt.close()


def plot_angles(df: pd.DataFrame, outdir: str, title_suffix: str = ""):
    """主角统计（均值/最大值）随时间演化（相对 baseline）"""
    plt.figure()
    plt.plot(df["t"], df["angle_mean_vs_base_deg"], marker="o", label="mean angle vs base (deg)")
    plt.plot(df["t"], df["angle_max_vs_base_deg"], marker="o", label="max angle vs base (deg)")
    plt.xlabel("checkpoint index (t)")
    plt.ylabel("degrees")
    plt.title(f"Principal Angles vs Baseline over Checkpoints{title_suffix}")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "angles_vs_base.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # 相邻对比
    if "angle_mean_vs_prev_deg" in df.columns:
        plt.figure()
        plt.plot(df["t"], df["angle_mean_vs_prev_deg"], marker="o", label="mean vs prev (deg)")
        plt.plot(df["t"], df["angle_max_vs_prev_deg"], marker="o", label="max vs prev (deg)")
        plt.xlabel("checkpoint index (t)")
        plt.ylabel("degrees")
        plt.title(f"Principal Angles vs Previous over Checkpoints{title_suffix}")
        plt.legend()
        plt.savefig(os.path.join(outdir, "angles_vs_prev.png"), dpi=160, bbox_inches="tight")
        plt.close()


def plot_overlap(df: pd.DataFrame, outdir: str, title_suffix: str = ""):
    """子空间重叠度随时间演化（相对 baseline 与相邻）"""
    plt.figure()
    plt.plot(df["t"], df["overlap_vs_base"], marker="o", label="overlap vs base")
    plt.xlabel("checkpoint index (t)")
    plt.ylabel("overlap [0,1]")
    plt.title(f"Subspace Overlap vs Baseline over Checkpoints{title_suffix}")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "overlap_vs_base.png"), dpi=160, bbox_inches="tight")
    plt.close()

    if "overlap_vs_prev" in df.columns:
        plt.figure()
        plt.plot(df["t"], df["overlap_vs_prev"], marker="o", label="overlap vs prev")
        plt.xlabel("checkpoint index (t)")
        plt.ylabel("overlap [0,1]")
        plt.title(f"Subspace Overlap vs Previous over Checkpoints{title_suffix}")
        plt.legend()
        plt.savefig(os.path.join(outdir, "overlap_vs_prev.png"), dpi=160, bbox_inches="tight")
        plt.close()


# ----------------------------
# 多层折线图（新增，关键）
# ----------------------------
def _num(x):
    return pd.to_numeric(x, errors="coerce")


def plot_angles_vs_base_multi(df_all: pd.DataFrame, outdir: str, title_suffix: str = ""):
    # mean
    plt.figure()
    for layer_idx, subdf in df_all.groupby("layer_idx"):
        x = _num(subdf["t"])
        y = _num(subdf["angle_mean_vs_base_deg"]).dropna()
        x = x.iloc[y.index]
        plt.plot(x, y, marker="o", label=f"layer{layer_idx}")
    plt.xlabel("checkpoint index (t)")
    plt.ylabel("mean angle vs base (deg)")
    plt.title(f"Principal Angles (mean) vs Baseline{title_suffix}")
    plt.legend(ncol=2, fontsize=8)
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "angles_vs_base_mean_all_layers.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # max
    plt.figure()
    for layer_idx, subdf in df_all.groupby("layer_idx"):
        x = _num(subdf["t"])
        y = _num(subdf["angle_max_vs_base_deg"]).dropna()
        x = x.iloc[y.index]
        plt.plot(x, y, marker="o", label=f"layer{layer_idx}")
    plt.xlabel("checkpoint index (t)")
    plt.ylabel("max angle vs base (deg)")
    plt.title(f"Principal Angles (max) vs Baseline{title_suffix}")
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(os.path.join(outdir, "angles_vs_base_max_all_layers.png"), dpi=160, bbox_inches="tight")
    plt.close()


def plot_angles_vs_prev_multi(df_all: pd.DataFrame, outdir: str, title_suffix: str = ""):
    # mean
    plt.figure()
    for layer_idx, subdf in df_all.groupby("layer_idx"):
        x = _num(subdf["t"])
        y = _num(subdf["angle_mean_vs_prev_deg"]).dropna()
        x = x.iloc[y.index]
        if len(y) == 0:
            continue
        plt.plot(x, y, marker="o", label=f"layer{layer_idx}")
    plt.xlabel("checkpoint index (t)")
    plt.ylabel("mean angle vs prev (deg)")
    plt.title(f"Principal Angles (mean) vs Previous{title_suffix}")
    plt.legend(ncol=2, fontsize=8)
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "angles_vs_prev_mean_all_layers.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # max
    plt.figure()
    for layer_idx, subdf in df_all.groupby("layer_idx"):
        x = _num(subdf["t"])
        y = _num(subdf["angle_max_vs_prev_deg"]).dropna()
        x = x.iloc[y.index]
        if len(y) == 0:
            continue
        plt.plot(x, y, marker="o", label=f"layer{layer_idx}")
    plt.xlabel("checkpoint index (t)")
    plt.ylabel("max angle vs prev (deg)")
    plt.title(f"Principal Angles (max) vs Previous{title_suffix}")
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(os.path.join(outdir, "angles_vs_prev_max_all_layers.png"), dpi=160, bbox_inches="tight")
    plt.close()


def plot_overlap_vs_base_multi(df_all: pd.DataFrame, outdir: str, title_suffix: str = ""):
    plt.figure()
    for layer_idx, subdf in df_all.groupby("layer_idx"):
        x = _num(subdf["t"])
        y = _num(subdf["overlap_vs_base"]).dropna()
        x = x.iloc[y.index]
        plt.plot(x, y, marker="o", label=f"layer{layer_idx}")
    plt.xlabel("checkpoint index (t)")
    plt.ylabel("overlap vs base [0,1]")
    plt.title(f"Subspace Overlap vs Baseline{title_suffix}")
    plt.legend(ncol=2, fontsize=8)
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "overlap_vs_base_all_layers.png"), dpi=160, bbox_inches="tight")
    plt.close()


def plot_overlap_vs_prev_multi(df_all: pd.DataFrame, outdir: str, title_suffix: str = ""):
    plt.figure()
    for layer_idx, subdf in df_all.groupby("layer_idx"):
        x = _num(subdf["t"])
        y = _num(subdf["overlap_vs_prev"]).dropna()
        x = x.iloc[y.index]
        if len(y) == 0:
            continue
        plt.plot(x, y, marker="o", label=f"layer{layer_idx}")
    plt.xlabel("checkpoint index (t)")
    plt.ylabel("overlap vs prev [0,1]")
    plt.title(f"Subspace Overlap vs Previous{title_suffix}")
    plt.legend(ncol=2, fontsize=8)
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "overlap_vs_prev_all_layers.png"), dpi=160, bbox_inches="tight")
    plt.close()


# ----------------------------
# 主流程（全层 + 新图）
# ----------------------------
if __name__ == "__main__":
    checkpoints = [
        "/data_net/models_for_all/llama2-13b-hf",
        "/data_net/fin_group/zyn/zhaomengcheng_outputs/uni1/llama-2-13b-clm_pt_full_uni_1_txt_5e-5_bf16/checkpoint-319",
        # "/data_net/fin_group/zyn/zhaomengcheng_outputs/uni1/llama-2-13b-clm_pt_full_uni_1_txt_5e-5_bf16/checkpoint-638",
        # "/data_net/fin_group/zyn/zhaomengcheng_outputs/uni1/llama-2-13b-clm_pt_full_uni_1_txt_5e-5_bf16/checkpoint-957",
        # "/data_net/fin_group/zyn/zhaomengcheng_outputs/uni1/llama-2-13b-clm_pt_full_uni_1_txt_5e-5_bf16/checkpoint-1276"
    ]

    which = "q_proj"
    k_subspace = 32
    topn_singular = 8

    # 先加载所有模型
    models = [load_llama_model(p) for p in checkpoints]
    num_layers = len(models[0].model.layers)

    # 所有层的结果拼接到一起
    dfs = []
    for layer_idx in tqdm(range(5)):
        print(f"Processing layer {layer_idx} ...")
        df, _ = compute_metrics_series(
            models,
            layer_idx=layer_idx,
            which=which,
            k_subspace=k_subspace,
            topn_singular=topn_singular
        )
        df["layer_idx"] = layer_idx
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    outdir = f"svd_vis_all_layers_{which}"
    os.makedirs(outdir, exist_ok=True)
    df_all.to_csv(os.path.join(outdir, "metrics_all_layers.csv"), index=False)

    # —— 新增：分别绘制 angles/overlap（vs base / vs prev）的“所有层折线图”
    suffix = f" ({which}, k={k_subspace})"
    plot_angles_vs_base_multi(df_all, outdir, title_suffix=suffix)
    plot_angles_vs_prev_multi(df_all, outdir, title_suffix=suffix)
    plot_overlap_vs_base_multi(df_all, outdir, title_suffix=suffix)
    plot_overlap_vs_prev_multi(df_all, outdir, title_suffix=suffix)

    print(f"Done. Results saved under: {outdir}")
