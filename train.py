# -*- coding: utf-8 -*-
"""
地聚物混凝土 28d 抗压强度预测 —— 简单多层感知机(MLP)

数据来源:《固废煤矸石细集料地聚物混凝土基本性能研究》(李庆轩)
特征为每立方米各组分用量(kg/m3)以及碱含量(%)、水玻璃模数。

用法:
    python train.py            # 使用默认数据训练并保存模型
    python train.py --epochs 3000 --hidden 32 16
"""
import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 输入特征(组分用量 + 碱含量 + 模数)与预测目标(28d抗压强度)
FEATURES = ["GGBS", "FA", "water_glass", "NaOH", "coal_gangue",
            "sand", "gravel", "water", "alkali_content", "modulus"]
TARGET = "fc_28d"


def set_seed(seed=42):
    """固定随机种子, 让结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """三层感知机: 输入 -> (隐藏层) -> 输出一个强度值。"""

    def __init__(self, in_dim, hidden=(32, 16), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_loader(x, y, batch_size=8, shuffle=True):
    """把 numpy 数组转成 PyTorch 的 DataLoader。"""
    ds = torch.utils.data.TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / n


def evaluate(model, loader, loss_fn):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb).numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return preds, trues


def fit(model, x, y, x_val, y_val, epochs, lr, patience):
    """训练模型, 用验证集做早停, 返回历史损失和最佳模型权重。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(x, y)
    val_loader = make_loader(x_val, y_val, shuffle=False)

    best_val, best_state, bad = float("inf"), None, 0
    hist = {"train": [], "val": []}
    for epoch in range(epochs):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn)
        # 用验证集 MSE 做早停依据
        vp, vt = evaluate(model, val_loader, loss_fn)
        val_mse = float(mean_squared_error(vt, vp))
        hist["train"].append(tr_loss)
        hist["val"].append(val_mse)
        if val_mse < best_val:
            best_val, bad = val_mse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return hist


def main():
    parser = argparse.ArgumentParser(description="地聚物混凝土强度预测 MLP 训练")
    parser.add_argument("--data", default="data/mix_data.csv")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    set_seed()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    X = df[FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    print(f"数据量: {len(df)} 条, 特征数: {len(FEATURES)}, 目标: {TARGET}")

    # 基线: 若所有样本都预测为均值, RMSE 是多少(R² 恒为 0)
    base_rmse = float(np.sqrt(mean_squared_error(y, np.full_like(y, y.mean()))))
    print(f"[基线] 全部预测均值 {y.mean():.1f} MPa 时 RMSE = {base_rmse:.3f} MPa (R²=0)")

    # ---------- K 折交叉验证, 评估模型泛化能力 ----------
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof_pred, oof_true = np.zeros_like(y), np.zeros_like(y)
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
        scaler = StandardScaler().fit(X[tr_idx])
        Xtr = scaler.transform(X[tr_idx])
        Xte = scaler.transform(X[te_idx])
        # 留出 20% 训练集做验证(早停用)
        n_val = max(1, int(len(tr_idx) * 0.2))
        val_idx = tr_idx[-n_val:]
        tr_idx2 = tr_idx[:-n_val]
        Xtr2, Xval = scaler.transform(X[tr_idx2]), scaler.transform(X[val_idx])

        model = MLP(len(FEATURES), hidden=tuple(args.hidden), dropout=args.dropout)
        fit(model, Xtr2, y[tr_idx2], Xval, y[val_idx],
            args.epochs, args.lr, args.patience)
        # 在测试折上用标准化后的数据预测
        with torch.no_grad():
            oof_pred[te_idx] = model(torch.tensor(Xte, dtype=torch.float32)).numpy()

    r2 = r2_score(y, oof_pred)
    rmse = np.sqrt(mean_squared_error(y, oof_pred))
    mae = mean_absolute_error(y, oof_pred)
    print("\n===== %d 折交叉验证结果 =====" % args.folds)
    print(f"R² = {r2:.4f}   RMSE = {rmse:.3f} MPa   MAE = {mae:.3f} MPa")
    print("(注: 样本仅 21 条, 交叉验证指标波动大、R² 为负属正常, 不代表代码有误; 见 README 局限说明)")

    # ---------- 用全部数据训练最终模型并保存 ----------
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    # 简单随机划分 80/20 做最终训练/验证
    idx = np.random.RandomState(42).permutation(len(X))
    n_val = max(1, int(len(X) * 0.2))
    tr_idx, val_idx = idx[:-n_val], idx[-n_val:]
    final_model = MLP(len(FEATURES), hidden=tuple(args.hidden), dropout=args.dropout)
    hist = fit(final_model, Xs[tr_idx], y[tr_idx], Xs[val_idx], y[val_idx],
               args.epochs, args.lr, args.patience)

    with torch.no_grad():
        final_pred = final_model(torch.tensor(Xs, dtype=torch.float32)).numpy()
    print("\n===== 最终模型(全量训练)在全部数据上的拟合 =====")
    print(f"R² = {r2_score(y, final_pred):.4f}   RMSE = {np.sqrt(mean_squared_error(y, final_pred)):.3f} MPa")
    print("(此处含训练数据, 数值偏乐观, 仅用于验证模型能学到规律)")

    # 保存模型与标准化器
    torch.save(final_model.state_dict(), os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    np.save(os.path.join(args.out, "oof_pred.npy"), oof_pred)
    np.save(os.path.join(args.out, "oof_true.npy"), y)

    # ---------- 画图 ----------
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["train"], label="train loss")
    plt.plot(hist["val"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("MSE loss")
    plt.title("Training curve"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y, oof_pred, c="steelblue", label="cross-val")
    lim = [min(y.min(), oof_pred.min()), max(y.max(), oof_pred.max())]
    plt.plot(lim, lim, "r--", label="y=x")
    plt.xlabel("Actual 28d strength (MPa)")
    plt.ylabel("Predicted 28d strength (MPa)")
    plt.title(f"Predicted vs Actual (R$^2$={r2:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "results.png"), dpi=120)
    print(f"\n结果已保存到 {args.out}/ (model.pt, scaler.pkl, results.png)")


if __name__ == "__main__":
    main()
