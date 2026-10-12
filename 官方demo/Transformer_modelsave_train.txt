# -*- coding: utf-8 -*-
"""Transformer 端到端 demo —— 训练侧脚本 (共享定义 + 从零训练并持久化)。

本文件承担两件事:
  1. 沉淀 **训练与推理共用** 的定义 (配置 / 模型结构 / 数据构建), 作为单一事实来源;
     配套 notebook 在推理时直接 `from transformer_train import ...` 复用, 避免两边漂移。
  2. 提供 `train_and_save(...)`: 在写死的训练区间上从零训练, 把
     **权重 + 标准化统计 + 结构超参** 一并保存到 `transformer_model.json` (纯文本)。

模型一律存为 **文本类文件 (JSON)**, 不使用 `.pt` 等二进制格式: state_dict 里的张量
会被转成 {dtype, shape, data(扁平 list)} 结构, 加载时按 dtype/shape 还原, 便于版本
管理、人工查阅与跨环境传输 (见 `save_model` / `load_model`)。

用法 (参赛者本地运行一次, 产物随 notebook 一起上传):
    python transformer_train.py
或在其它脚本/notebook 中:
    from transformer_train import train_and_save
    train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"})

公榜阶段平台不会重训, 直接加载该文件做推理 (见 notebook 的 `main`);
私榜阶段平台用 `train_and_save` 在隔离环境从零重训, 故训练逻辑需保持可复现 (固定随机种子)。
"""
import os
import json
import time

import numpy as np
import pandas as pd
import dai
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import structlog

logger = structlog.get_logger()

# 训练好的模型保存路径; 参赛者本地训练后, 把该文件随 notebook 一并上传
# 平台限制只能提交文本类文件, 故存为 JSON (而非 torch 的 .pt 二进制)
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "transformer_model.json")

# ---------- 配置 (写死, 不随平台入参变化) ----------
TRAIN_START, TRAIN_END = "2020-01-01", "2020-12-31 23:59:59"  # 训练区间写死, 切勿用平台注入的测试区间训练
SEQ_LEN = 64                  # 每个 30 分钟截面回看的原始 1m bar 数
EPOCHS, BATCH, LR, SEED = 5, 512, 1e-3, 42
MAX_TRAIN_INSTRUMENTS = 200   # demo 限制训练标的数控制时长, 正式可放开

# 直接使用 bar1m 原始字段，不构造派生特征，只做按字段标准化。
FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "amount",
    "bid_price1", "ask_price1", "bid_volume1", "ask_volume1",
]
N_FEAT = len(FEATURE_COLS)
LABEL_TABLE = "cpt_jyc_2026_vwap"
SECTION_TABLE = "cpt_jyc_2026_instruments"

# 模型结构超参 (训练与推理必须一致, 会一并存入权重文件供推理端重建模型)
MODEL_CFG = dict(n_feat=N_FEAT, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN)
DATA_ALIGNMENT = "30m_section_uses_bar_end_le_section_and_vwap_return"


# ---------- 模型: 单条 Transformer 编码 -> 池化 -> 回归头 ----------
class StockTransformer(nn.Module):
    def __init__(self, n_feat, d_model=64, nhead=4, nlayers=2, dim_ff=128, seq_len=SEQ_LEN):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)                  # 每个 bar -> token 向量
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))  # 可学习位置编码
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, 0.1,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):                                       # (B, L, N_FEAT) -> (B,)
        h = self.encoder(self.proj(x) + self.pos).mean(dim=1)
        return self.head(h).squeeze(-1)


def pool(sd, ed):
    """区间内中证 1000 成分股代码。"""
    df = dai.query(f"SELECT DISTINCT instrument FROM {SECTION_TABLE}",
                   filters={"date": [sd, ed]}).df()
    return sorted(df["instrument"].dropna().unique().tolist())


# ---------- 数据: 原始 1m bar -> 30 分钟截面，只做标准化 ----------
def build_dataset(table, sd, ed, mode, instruments, stats=None):
    """构建比赛的 30 分钟截面样本。

    截面时间来自 cpt_jyc_2026_instruments；训练标签直接使用
    cpt_jyc_2026_vwap.vwap_return。每个截面只使用 date <= 截面时间的原始
    1 分钟 bar，避免未来数据泄漏。
    """
    t0 = time.time()
    sd_ts = pd.to_datetime(sd)
    buf = (sd_ts - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    sql = f"SELECT date, instrument, {', '.join(FEATURE_COLS)} FROM {table} ORDER BY instrument, date"
    df = dai.query(sql, filters={"date": [buf, ed], "instrument": instruments}).df()
    df["date"] = pd.to_datetime(df["date"])

    # 以比赛股票池中的 8 个日内时点作为样本主键。
    section_df = dai.query(
        f"SELECT date, instrument FROM {SECTION_TABLE} ORDER BY instrument, date",
        filters={"date": [sd, ed], "instrument": instruments},
    ).df()
    section_df["date"] = pd.to_datetime(section_df["date"])
    section_df = section_df.drop_duplicates(["date", "instrument"])

    if mode == "train":
        labels = dai.query(
            f"SELECT date, instrument, vwap_return FROM {LABEL_TABLE}",
            filters={"date": [sd, ed], "instrument": instruments},
        ).df()
        labels["date"] = pd.to_datetime(labels["date"])
        labels = labels.replace([np.inf, -np.inf], np.nan).dropna(subset=["vwap_return"])
        section_df = section_df.merge(
            labels, on=["date", "instrument"], how="inner", validate="one_to_one"
        )

    wins, ys, sample_keys = [], [], []
    section_groups = {
        ins: sub.sort_values("date")
        for ins, sub in section_df.groupby("instrument", sort=False)
    }
    for ins, bars in df.groupby("instrument", sort=False):
        sections = section_groups.get(ins)
        if sections is None or len(bars) < SEQ_LEN:
            continue
        bars = bars.sort_values("date")
        bar_time = bars["date"].to_numpy(dtype="datetime64[ns]")
        feats = bars[FEATURE_COLS].to_numpy(np.float32)
        for row in sections.itertuples(index=False):
            section_time = pd.Timestamp(row.date)
            end = np.searchsorted(bar_time, section_time.to_datetime64(), side="right")
            if end < SEQ_LEN:
                continue
            window = feats[end - SEQ_LEN:end]
            if pd.Timestamp(bar_time[end - 1]) > section_time:
                raise RuntimeError(f"特征时点泄漏: {ins}, {section_time}")
            wins.append(window)
            ys.append(np.float32(row.vwap_return) if mode == "train" else np.float32(0.0))
            sample_keys.append((section_time, ins))
    if not sample_keys:
        raise RuntimeError(f"build_dataset 无样本 (mode={mode}, {sd}~{ed})")

    X = np.stack(wins).astype(np.float32)                       # (N, SEQ_LEN, N_FEAT)
    X[~np.isfinite(X)] = np.nan
    if stats is None:                                           # 只在训练集上计算
        flat = X.reshape(-1, N_FEAT)
        stats = (np.nanmean(flat, 0).astype(np.float32),
                 (np.nanstd(flat, 0) + 1e-6).astype(np.float32))
    m, s = stats
    X = np.nan_to_num((X - m) / s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    logger.info(f"{mode} 集构建完成", samples=len(sample_keys), elapsed=round(time.time() - t0, 2))
    if mode == "train":
        return X, np.array(ys, np.float32), None, stats
    return X, None, pd.DataFrame(sample_keys, columns=["date", "instrument"]), stats


# ==== 模型存/读: 一律用文本类文件 (JSON), 不使用 .pt 等二进制 ====
def save_model(ckpt, model_path=MODEL_PATH):
    """把 checkpoint 存成 JSON 文本文件。

    state_dict 里每个张量转成 {dtype, shape, data(扁平 list)}, 其余字段 (结构超参 /
    标准化统计等) 原样写入; 加载时用 load_model 按 dtype/shape 还原。"""
    sd = ckpt["state_dict"]
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        tensors[k] = {
            "dtype": str(t.dtype).replace("torch.", ""),   # 如 'float32'
            "shape": list(t.shape),
            "data": t.reshape(-1).tolist(),                # 扁平存, 加载时按 shape 还原
        }
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    """读取 save_model 写出的 JSON, 把 state_dict 还原为张量 dict。

    返回结构与原 torch.load(...) 的 checkpoint 一致 (state_dict 为 {name: Tensor})。"""
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt


# ==== 训练并持久化 (参赛者本地运行一次, 产物随 notebook 一起上传) ====
def train_and_save(datasources, model_path=MODEL_PATH):
    """在写死的训练区间上从零训练, 把 权重 + 标准化统计 + 结构超参 一并存盘。

    公榜阶段平台不会重训, 直接加载该文件做推理; 私榜阶段平台用本函数从零重训,
    故训练逻辑需保持可复现 (固定随机种子)。"""
    table = datasources["bar1m"]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(device), table=table)

    # ---------- 构建训练集 (写死训练区间) ----------
    logger.info("构建训练集", start=TRAIN_START, end=TRAIN_END)
    Xtr, ytr, _, stats = build_dataset(
        table, TRAIN_START, TRAIN_END, "train",
        pool(TRAIN_START, TRAIN_END)[:MAX_TRAIN_INSTRUMENTS])

    # ---------- 从零训练 ----------
    model = StockTransformer(**MODEL_CFG).to(device)
    logger.info("可训练参数量", n_params=sum(p.numel() for p in model.parameters()))
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True, pin_memory=(device.type == "cuda"))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(EPOCHS):
        t, tot, nb = time.time(), 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        logger.info("epoch 完成", epoch=ep + 1, mse=round(tot / max(nb, 1), 8),
                    elapsed=round(time.time() - t, 2))

    # ---------- 持久化: 权重 + 统计 + 结构超参 (推理端据此重建并复用) ----------
    # 一律存为文本类文件 (JSON), 张量在 save_model 内转成 {dtype, shape, data}
    mean, std = stats
    save_model({
        "state_dict": model.state_dict(),
        "model_cfg": MODEL_CFG,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "data_alignment": DATA_ALIGNMENT,
        "mean": np.asarray(mean, np.float32).tolist(),         # 存为 list, 加载更稳健
        "std": np.asarray(std, np.float32).tolist(),
    }, model_path)
    logger.info("模型已保存, 请随 notebook 一并上传", path=model_path)
    return model_path


if __name__ == "__main__":
    # 本地训练入口: 在写死的训练区间上从零训练并保存 transformer_model.json
    datasources = {"bar1m": "cpt_jyc_2026_stock_bar1m"}
    train_and_save(datasources)
