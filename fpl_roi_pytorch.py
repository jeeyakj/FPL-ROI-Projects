# fpl_roi_pytorch.py
# Train a PyTorch model to predict FPL expected points and compute ROI with uncertainty

import os
import math
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# Config
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# Use GPU if available

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Utilities
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def print_header(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

 
# Data assumptions and schema
 
"""
Expected CSV structure (you can adapt these names to your dataset):

Columns (example):
- player_id (int or str)
- player_name (str)
- position (str: 'GK', 'DEF', 'MID', 'FWD')
- team (str)
- opponent_team (str)
- was_home (bool or 0/1)
- price (float; current or at GW)
- minutes (int)
- goals (int)
- assists (int)
- xG (float)
- xA (float)
- shots (int)
- key_passes (int)
- ict_index (float)
- bps (int)
- fixture_difficulty (float or int)
- gw (int)
- season (str or int)
- points_next_gw (float)  # Target: points scored in the next gameweek

If you don't have points_next_gw, you can approximate by shifting per player:
- For each player_id, target = points of the next GW
"""

 
# Feature engineering
def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Returns:
      df: with clean features and target
      feature_cols: list of columns used as model input
      target_col: target column name for regression
    """
    # Basic cleaning
    df = df.copy()
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure required columns exist (adapt as needed)
    required = [
        "player_id", "player_name", "position", "team", "opponent_team",
        "was_home", "price", "minutes", "goals", "assists", "xg", "xa",
        "shots", "key_passes", "ict_index", "bps", "fixture_difficulty",
        "gw", "season"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # If target doesn't exist, create it by shifting within player_id
    if "points_next_gw" not in df.columns:
        if "total_points" not in df.columns:
            raise ValueError("Need 'points_next_gw' or 'total_points' to build a target via shift.")
        df = df.sort_values(["player_id", "season", "gw"]).copy()
        df["points_next_gw"] = df.groupby("player_id")["total_points"].shift(-1)
        # Drop last GW for each player where next GW doesn't exist
        df = df.dropna(subset=["points_next_gw"])

    # Convert categoricals to one-hot
    # Position one-hot
    pos_dummies = pd.get_dummies(df["position"], prefix="pos")
    team_dummies = pd.get_dummies(df["team"], prefix="team", dummy_na=False)
    opp_dummies = pd.get_dummies(df["opponent_team"], prefix="opp", dummy_na=False)

    # was_home to int
    df["was_home"] = df["was_home"].astype(int)

    # Continuous inputs (add/remove)
    numeric_features = [
        "minutes", "goals", "assists", "xg", "xa",
        "shots", "key_passes", "ict_index", "bps", "fixture_difficulty"
    ]

    # Clean numeric features
    for col in numeric_features + ["price", "points_next_gw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_features + ["price", "points_next_gw"])

    # Concatenate features
    feature_df = pd.concat([df[numeric_features + ["was_home"]], pos_dummies, team_dummies, opp_dummies], axis=1)

    feature_cols = list(feature_df.columns)
    target_col = "points_next_gw"

    # Attach the engineered features back
    df_features = pd.concat([df[["player_id", "player_name", "price", "season", "gw", target_col]], feature_df], axis=1)

    return df_features, feature_cols, target_col

 
# Standardization helpers
 
def compute_standardization(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    mean = df[feature_cols].mean()
    std = df[feature_cols].std().replace(0, 1.0)  # avoid division by zero
    return mean, std
# Apply standardization to a DataFrame using provided mean and std
# Note: we can also clip extreme z-scores if desired to reduce outlier impact
# This function can be used for both train and validation/test sets (use train mean/std for val/test)
# We return a new DataFrame with standardized features, keeping meta columns intact.
def apply_standardization(df: pd.DataFrame, feature_cols: List[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = (df[feature_cols] - mean) / std
    # Clip extreme z-scores if helpful
    df[feature_cols] = df[feature_cols].clip(-5, 5)
    return df

 
# Dataset and DataLoaders
 
class FPLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)
        self.meta = df[["player_id", "player_name", "price", "season", "gw"]].reset_index(drop=True)
# The meta DataFrame allows us to keep track of player info for each row, which is useful for output and analysis.
# The __getitem__ method returns the features, target, and meta info for each index, which can be used in the training loop and when making predictions.
# Note: we could also return the original index if needed for merging predictions back to the original DataFrame.
# The DataLoader will handle batching and shuffling for us during training and evaluation.
# We ensure that the meta info is aligned with the features and target by resetting the index after selecting the relevant columns.
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.meta.iloc[idx].to_dict()

 
# Model
 
class FPLMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU()) 
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))  # output: expected points
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

 
# Training and evaluation
 
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8
) -> Dict[str, float]:
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    # Early stopping variables
    # We track the best validation MSE and save the model state when we see an improvement. If we go for 'patience' epochs without improvement, we stop training.
    # The best model state is stored in CPU memory to avoid GPU memory issues, and we load it back into the model at the end of training.
    # We use a small threshold (1e-6) to determine if the improvement is significant enough to reset the patience counter.
    # This simple early stopping mechanism helps prevent overfitting and saves time by not training for unnecessary epochs once the model has converged.
    # Note: for a more robust implementation, we could also save the best model to disk and load it back, especially if the model is large.

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for X, y, _ in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y, _ in val_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                preds = model(X)
                loss = criterion(preds, y)
                val_losses.append(loss.item())
                # We could also collect predictions and targets here for more detailed analysis (e.g., scatter plot, residuals) after training.
                # This would allow us to see how well the model is performing across different ranges of points and identify any systematic errors.

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")

        print(f"Epoch {epoch:03d} | train MSE: {avg_train:.4f} | val MSE: {avg_val:.4f}")

        # Early stopping
        if avg_val < best_val - 1e-6:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}. Best val MSE: {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mse": best_val}

def predict_mc_dropout(model: nn.Module, X: torch.Tensor, passes: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout: run multiple forward passes with dropout enabled to estimate mean and std.
    """
    # We set the model to train mode to enable dropout during inference, which allows us to capture the uncertainty in the predictions. By running multiple forward passes, we can compute the mean and standard deviation of the predictions, which gives us an estimate of the expected points and the uncertainty around that estimate.
    # Note: this is a simple way to get uncertainty estimates without needing a full Bayesian neural network. The standard deviation can be interpreted as the model's confidence in its predictions, with higher values indicating more uncertainty.
    # We can adjust the number of passes to balance between accuracy of the uncertainty estimate and computation time. More passes will give a better estimate but will take longer to compute.
    model.train()  # important: enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(passes):
            out = model(X.to(DEVICE)).cpu().numpy()
            preds.append(out)
    preds = np.stack(preds, axis=0)  # [passes, N, 1]
    mean = preds.mean(axis=0).squeeze(1)
    std = preds.std(axis=0).squeeze(1)
    return mean, std

 
# Train/val split
 
def time_aware_split(df: pd.DataFrame, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple time-aware split: use earlier rows for train, later rows for validation.
    For better rigor, split by season/GW to avoid leakage.
    """
    df = df.sort_values(["season", "gw"])
    n = len(df)
    split = int((1 - val_ratio) * n)
    return df.iloc[:split].copy(), df.iloc[split:].copy()

 
# Main pipeline
 
def run_pipeline(
    csv_path: str,
    output_path: str = "fpl_roi_predictions.csv",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 128,
    dropout: float = 0.2,
    mc_passes: int = 100
):
    # Load data
    print_header("Loading data")
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded {len(df_raw)} rows from {csv_path}")

    print_header("Engineering features")
    df_features, feature_cols, target_col = engineer_features(df_raw)
    print(f"Feature columns: {len(feature_cols)} | Target: {target_col}")
    print(f"Examples:\n{df_features.head(3)}")

    print_header("Splitting train/validation")
    df_train, df_val = time_aware_split(df_features, val_ratio=0.2)
    print(f"Train rows: {len(df_train)} | Val rows: {len(df_val)}")

    print_header("Standardizing features")
    mean, std = compute_standardization(df_train, feature_cols)
    df_train_std = apply_standardization(df_train, feature_cols, mean, std)
    df_val_std = apply_standardization(df_val, feature_cols, mean, std)

    train_ds = FPLDataset(df_train_std, feature_cols, target_col)
    val_ds = FPLDataset(df_val_std, feature_cols, target_col)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    print_header("Building model")
    model = FPLMLP(input_dim=len(feature_cols), hidden_dims=[128, 64, 32], dropout=dropout)
    print(model)

    print_header("Training")
    stats = train(model, train_loader, val_loader, epochs=epochs, lr=lr, weight_decay=weight_decay, patience=8)
    print(f"Best validation MSE: {stats['best_val_mse']:.4f}")

    # Save model (optional)
    torch.save({"state_dict": model.state_dict(), "mean": mean.to_dict(), "std": std.to_dict(), "feature_cols": feature_cols}, "fpl_model.pt")
    print("Model saved to fpl_model.pt")

    print_header("Predicting with uncertainty (Monte Carlo Dropout)")
    # Prepare full standardized dataset for predictions (use latest rows or full val set)
    df_all_std = pd.concat([df_train_std, df_val_std], axis=0).reset_index(drop=True)
    all_ds = FPLDataset(df_all_std, feature_cols, target_col)
    all_loader = DataLoader(all_ds, batch_size=batch_size, shuffle=False)

    # Collect predictions
    preds_mean_list = []
    preds_std_list = []
    metas = []
    with torch.no_grad():
        for X, y, meta in all_loader:
            mean_batch, std_batch = predict_mc_dropout(model, X, passes=mc_passes)
            preds_mean_list.append(mean_batch)
            preds_std_list.append(std_batch)
            metas.extend(list(meta))

    preds_mean = np.concatenate(preds_mean_list, axis=0)
    preds_std = np.concatenate(preds_std_list, axis=0)

    # Compute ROI and confidence intervals
    # ROI = predicted_points / price
    out_rows = []
    for i, m in enumerate(metas):
        price = safe_float(m["price"], default=1.0)
        expected_points = preds_mean[i]
        uncertainty_points = preds_std[i]
        roi = expected_points / price if price > 0 else np.nan

        # 95% normal approx interval for points and ROI
        ci_points_low = expected_points - 1.96 * uncertainty_points
        ci_points_high = expected_points + 1.96 * uncertainty_points
        roi_low = ci_points_low / price if price > 0 else np.nan
        roi_high = ci_points_high / price if price > 0 else np.nan

        out_rows.append({
            "player_id": m["player_id"],
            "player_name": m["player_name"],
            "season": m["season"],
            "gw": m["gw"],
            "price": price,
            "pred_points_mean": float(expected_points),
            "pred_points_std": float(uncertainty_points),
            "roi_mean": float(roi),
            "roi_95_low": float(roi_low),
            "roi_95_high": float(roi_high)
        })

    df_out = pd.DataFrame(out_rows)

    print_header("Top players by expected ROI (mean)")
    top = df_out.sort_values("roi_mean", ascending=False).head(20)
    print(top[["player_name", "season", "gw", "price", "pred_points_mean", "pred_points_std", "roi_mean", "roi_95_low", "roi_95_high"]])

    print_header(f"Writing predictions to {output_path}")
    df_out.to_csv(output_path, index=False)
    print(f"Done. Rows: {len(df_out)}")

 
# CLI
 
def parse_args():
    parser = argparse.ArgumentParser(description="FPL ROI Calculator with PyTorch")
    parser.add_argument("--csv", type=str, required=True, help="Path to historical FPL CSV data")
    parser.add_argument("--out", type=str, default="fpl_roi_predictions.csv", help="Output CSV path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mc", type=int, default=100, help="Monte Carlo dropout passes")
    return parser.parse_args()
# Example usage:
# python fpl_roi_pytorch.py --csv historical_fpl_data.csv --out fpl_roi_predictions.csv --epochs 50 --lr 0.001 --wd 1e-5 --batch 128 --dropout 0.2 --mc 100
# This will train the model on the provided historical data, compute expected points and ROI with uncertainty estimates, and save the results to a CSV file.

if __name__ == "__main__":
    set_seed(SEED)
    args = parse_args()
    run_pipeline(
        csv_path=args.csv,
        output_path=args.out,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        batch_size=args.batch,
        dropout=args.dropout,
        mc_passes=args.mc
    )
    # After running, you can analyze the output CSV to see which players have the best expected ROI and how confident the model is in those predictions. You can also compare the predicted points to actual points in future GWs to evaluate model performance over time.
    # Note: this is a basic implementation and can be further improved with more sophisticated feature engineering, model architectures, hyperparameter tuning, and evaluation metrics.