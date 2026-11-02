"""Neural travel-time model with a train/validation/test protocol.

The test split is touched only after checkpoint selection is complete.
"""
from collections import defaultdict
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

VAN_FACTOR = 1.2


class TripNN(nn.Module):
    def __init__(self, n_zone=265):
        super().__init__()
        self.pu = nn.Embedding(n_zone + 1, 16)
        self.do = nn.Embedding(n_zone + 1, 16)
        self.h = nn.Embedding(24, 4)
        self.w = nn.Embedding(7, 3)
        self.mlp = nn.Sequential(
            nn.Linear(16 + 16 + 4 + 3 + 1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, pu, do, hour, weekday, distance):
        features = torch.cat([
            self.pu(pu), self.do(do), self.h(hour), self.w(weekday),
            distance.unsqueeze(1),
        ], dim=1)
        return self.mlp(features).squeeze(1)


def get_tensors(df, mu, sd):
    return (
        torch.as_tensor(df.PULocationID.to_numpy(), dtype=torch.long),
        torch.as_tensor(df.DOLocationID.to_numpy(), dtype=torch.long),
        torch.as_tensor(df.hour.to_numpy(), dtype=torch.long),
        torch.as_tensor(df.weekday.to_numpy(), dtype=torch.long),
        torch.as_tensor(((df.pair_mean_dist - mu) / sd).to_numpy(), dtype=torch.float32),
        torch.as_tensor(df.duration.to_numpy(), dtype=torch.float32),
    )


def train(epochs=30, validation_fraction=0.10, seed=42):
    """Train on the training partition, select on validation, evaluate once on test."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    torch.manual_seed(seed)
    np.random.seed(seed)

    full_train = pd.read_parquet("train_20zones.parquet")
    test = pd.read_parquet("test_20zones.parquet")
    if len(full_train) < 2 or len(test) == 0:
        raise ValueError("Training and test parquet files must contain enough rows")

    shuffled = full_train.sample(frac=1.0, random_state=seed)
    n_val = max(1, int(round(len(shuffled) * validation_fraction)))
    validation = shuffled.iloc[:n_val].copy()
    fit = shuffled.iloc[n_val:].copy()
    if fit.empty:
        raise ValueError("Validation split leaves no training rows")

    # Fit normalization statistics on training rows only.
    mu = float(fit.pair_mean_dist.mean())
    sd = float(fit.pair_mean_dist.std())
    if not np.isfinite(sd) or sd < 1e-6:
        sd = 1.0

    fit_t = get_tensors(fit, mu, sd)
    val_t = get_tensors(validation, mu, sd)
    test_t = get_tensors(test, mu, sd)

    model = TripNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    os.makedirs("results", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(fit_t[0]))
        for start in range(0, len(permutation), 2048):
            batch = permutation[start:start + 2048]
            prediction = model(*[tensor[batch] for tensor in fit_t[:5]])
            loss = loss_fn(prediction, fit_t[5][batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mae = loss_fn(model(*val_t[:5]), val_t[5]).item()
        print(f"epoch={epoch + 1} validation_mae={val_mae:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch + 1
            patience = 0
            torch.save({
                "state_dict": model.state_dict(),
                "mu": mu,
                "sd": sd,
                "best_epoch": best_epoch,
                "validation_mae": best_val,
                "seed": seed,
            }, "nn.pt")
        else:
            patience += 1
            if patience >= 4:
                break

    checkpoint = torch.load("nn.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"selected_epoch={best_epoch} validation_mae={best_val:.4f}")

    # Test metrics are computed only after model selection is finished.
    metrics = evaluate_baselines(fit, test, test_t, model)
    export_matrix(model=model, checkpoint=checkpoint)
    return metrics


def evaluate_baselines(train_df, test_df, test_tensors, model=None):
    os.makedirs("results", exist_ok=True)
    grouped = train_df.groupby(
        ["PULocationID", "DOLocationID", "hour"]
    ).duration.mean()
    global_mean = float(train_df.duration.mean())
    pred_avg = [
        float(grouped.get((r.PULocationID, r.DOLocationID, r.hour), global_mean))
        for r in test_df.itertuples()
    ]
    # This baseline uses observed test trip distance; it is an oracle-style
    # comparison, not a feature available to a pre-trip dispatcher.
    pred_distance = [r.trip_distance / 18.0 * 60.0 for r in test_df.itertuples()]

    if model is None:
        checkpoint = torch.load("nn.pt", map_location="cpu", weights_only=False)
        model = TripNN()
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        pred_nn = model(*test_tensors[:5]).cpu().numpy()

    actual = test_df.duration.to_numpy()
    def mae(prediction):
        return float(np.mean(np.abs(actual - prediction)))
    def rmse(prediction):
        return float(np.sqrt(np.mean((actual - prediction) ** 2)))

    result = pd.DataFrame([
        {"model": "NN", "MAE": mae(pred_nn), "RMSE": rmse(pred_nn)},
        {"model": "hist-avg", "MAE": mae(pred_avg), "RMSE": rmse(pred_avg)},
        {"model": "distance", "MAE": mae(pred_distance), "RMSE": rmse(pred_distance)},
    ])
    result.to_csv("results/nn_vs_baseline.csv", index=False)
    print(result.to_string(index=False))
    return result


def export_matrix(model=None, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load("nn.pt", map_location="cpu", weights_only=False)
    if model is None:
        model = TripNN()
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mu, sd = float(checkpoint["mu"]), float(checkpoint["sd"])
    with open("top20.json", encoding="utf-8") as handle:
        top20 = json.load(handle)
    pair = pd.read_csv("pair_mean_dist_train.csv")
    distance_map = {
        (int(row.PULocationID), int(row.DOLocationID)): float(row.pair_mean_dist)
        for row in pair.itertuples()
    }
    global_distance = float(pair.pair_mean_dist.mean())
    matrix = np.zeros((20, 20, 24), dtype=np.float32)
    with torch.no_grad():
        for i, pickup in enumerate(top20):
            for j, dropoff in enumerate(top20):
                normalized_distance = (distance_map.get((pickup, dropoff), global_distance) - mu) / sd
                for hour in range(24):
                    inputs = (
                        torch.tensor([pickup]), torch.tensor([dropoff]),
                        torch.tensor([hour]), torch.tensor([2]),
                        torch.tensor([normalized_distance], dtype=torch.float32),
                    )
                    matrix[i, j, hour] = float(model(*inputs).item()) * VAN_FACTOR
    np.save("time_matrix_20x20x24.npy", matrix)
    return matrix


if __name__ == "__main__":
    train()
