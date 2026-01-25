# ============================================================
# Higher-order interaction benchmark (DPWGAN):
# includes a sweep over beta_int values
#
# For each beta_int:
#   - simulate once (SIM_SEED_BASE + offset)
#   - split once (SPLIT_SEED_BASE + offset)
#   - compute REAL baseline metrics
#   - run DPWGAN synth over eps_grid x reps
#   - evaluate synthetic -> REAL holdout
#   - save:
#       * per-run results CSV
#       * grouped summary CSV (by beta_int, epsilon)
#
# SIM_SETUP: "constant" (equicorr) or "nonconstant" (rho12/rho13/rho23)
#
# Notes:
# - Uses repo DPWGAN training loop (gradient noise via sigma + weight clipping).
# - Treats V1..V4 as continuous; standardize using TRAIN stats.
# - "epsilon" is an index into a user-provided mapping EPS -> SIGMA / EPOCHS.
# ============================================================

import os
import math
import random
import logging
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import torch
import torch.nn as nn

# Adjust import to match your project
from dpwgan import DPWGAN


# -------------------------------
# 0) Simulate benchmark data
# -------------------------------
def simulate_benchmark_constant(
    n: int = 20_000,
    rho: float = 0.7,
    beta_main: Tuple[float, float, float] = (0.6, 0.6, 0.6),
    beta_int: float = 3.0,
    eps_sd: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Sigma = np.full((3, 3), float(rho), dtype=float)
    np.fill_diagonal(Sigma, 1.0)

    X = rng.multivariate_normal(mean=[0.0, 0.0, 0.0], cov=Sigma, size=int(n))
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    x4 = (
        beta_main[0] * x1
        + beta_main[1] * x2
        + beta_main[2] * x3
        + float(beta_int) * (x1 * x2 * x3)
        + rng.normal(0.0, float(eps_sd), size=int(n))
    )
    return pd.DataFrame({"V1": x1, "V2": x2, "V3": x3, "V4": x4})


def simulate_benchmark_nonconstant(
    n: int = 20_000,
    rho12: float = 0.70,
    rho13: float = 0.50,
    rho23: float = 0.30,
    beta_main: Tuple[float, float, float] = (0.6, 0.6, 0.6),
    beta_int: float = 3.0,
    eps_sd: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Sigma = np.array(
        [
            [1.0, float(rho12), float(rho13)],
            [float(rho12), 1.0, float(rho23)],
            [float(rho13), float(rho23), 1.0],
        ],
        dtype=float,
    )

    Sigma = (Sigma + Sigma.T) / 2.0
    eigmin = float(np.min(np.linalg.eigvalsh(Sigma)))
    if eigmin <= 1e-10:
        Sigma = Sigma + (abs(eigmin) + 1e-6) * np.eye(3)

    X = rng.multivariate_normal(mean=[0.0, 0.0, 0.0], cov=Sigma, size=int(n))
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    x4 = (
        beta_main[0] * x1
        + beta_main[1] * x2
        + beta_main[2] * x3
        + float(beta_int) * (x1 * x2 * x3)
        + rng.normal(0.0, float(eps_sd), size=int(n))
    )
    return pd.DataFrame({"V1": x1, "V2": x2, "V3": x3, "V4": x4})


# -------------------------------
# 1) Split train/holdout once
# -------------------------------
def train_test_split(df: pd.DataFrame, train_frac: float = 0.7, seed: int = 1234):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_train = int(math.floor(train_frac * len(df)))
    train_idx = idx[:n_train]
    hold_idx = idx[n_train:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[hold_idx].reset_index(drop=True)


# -------------------------------
# 2) Metrics helpers
# -------------------------------
def rmse(y, yhat) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def r2_oos(y, yhat) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def extract_three_way_stats(fit, term: str = "V1:V2:V3"):
    if term not in fit.params.index:
        return float("nan"), float("nan"), float("nan")
    return float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])


# -------------------------------
# 3) Repro helpers
# -------------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------
# 4) Standardize (train stats only)
# -------------------------------
def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd


def standardize_invert(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return Z * sd + mu


# -------------------------------
# 5) Simple MLP generator/discriminator for d=4
# -------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim: int, out_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


# -------------------------------
# 6) DPWGAN synth wrapper (continuous)
# -------------------------------
def dpwgan_synthesize(
    X_train: pd.DataFrame,
    *,
    seed: int,
    sigma: float,
    epochs: int,
    noise_dim: int = 32,
    hidden_dim: int = 64,
    n_critics: int = 5,
    learning_rate: float = 1e-4,
    weight_clip: float = 0.1,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Trains DPWGAN on standardized continuous data and returns synthetic DataFrame (unstandardized).
    """
    set_all_seeds(seed)

    cols = list(X_train.columns)
    X_np = X_train.to_numpy(dtype=float)

    mu, sd = standardize_fit(X_np)
    X_z = standardize_apply(X_np, mu, sd)

    data = torch.tensor(X_z, dtype=torch.float32, device=device)

    G = Generator(noise_dim=noise_dim, out_dim=data.shape[1], hidden_dim=hidden_dim).to(device)
    D = Discriminator(in_dim=data.shape[1], hidden_dim=hidden_dim).to(device)

    noise_fn = lambda n: torch.randn(int(n), int(noise_dim), device=device)
    gan = DPWGAN(generator=G, discriminator=D, noise_function=noise_fn)

    gan.train(
        data=data,
        epochs=int(epochs),
        n_critics=int(n_critics),
        batch_size=min(128, data.shape[0]),
        learning_rate=float(learning_rate),
        sigma=float(sigma) if sigma is not None else None,
        weight_clip=float(weight_clip),
    )

    with torch.no_grad():
        synth_z = gan.generate(int(data.shape[0])).detach().cpu().numpy()

    synth = standardize_invert(synth_z, mu, sd)
    return pd.DataFrame(synth, columns=cols)


# -------------------------------
# 7) Main experiment with beta_int sweep
# -------------------------------
def main():
    logging.basicConfig(level=logging.INFO)

    # -------------------------------
    # USER SWITCHES (match MST script)
    # -------------------------------
    SIM_SETUP = "constant"  # "constant" or "nonconstant"
    eps_grid = [1, 2, 3, 4, 5]
    reps = list(range(1, 10))
    beta_int_grid = [0, 0.5, 1, 2, 3, 5]

    # Data/noise settings (match MST script)
    N_SIM = 20_000
    N_SIM_NC = 20_000
    RHO_EQ = 0.7
    RHO12, RHO13, RHO23 = 0.70, 0.50, 0.30
    EPS_SD_CONST = 0.5
    EPS_SD_NC = 1.0

    SIM_SEED_BASE = 123
    SPLIT_SEED_BASE = 1234

    # DPWGAN hyperparams
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NOISE_DIM = 32
    HIDDEN_DIM = 64
    N_CRITICS = 5
    LR = 1e-4
    WEIGHT_CLIP = 0.1

    # You provide this mapping
    SIGMA_BY_EPS = {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0}
    EPOCHS_BY_EPS = {1: 20, 2: 74, 3: 143, 4: 244, 5: 383}

    OUT_ROOT = f"higher_order_dpwgan_sweep_{SIM_SETUP}"
    os.makedirs(OUT_ROOT, exist_ok=True)

    rows: List[Dict] = []
    cols = ["V1", "V2", "V3", "V4"]

    for b_idx, beta_int in enumerate(beta_int_grid, start=1):
        SIM_SEED = SIM_SEED_BASE + 10_000 * b_idx
        SPLIT_SEED = SPLIT_SEED_BASE + 10_000 * b_idx

        # ---- simulate ----
        if SIM_SETUP == "constant":
            df = simulate_benchmark_constant(
                n=N_SIM,
                rho=RHO_EQ,
                beta_int=float(beta_int),
                eps_sd=EPS_SD_CONST,
                seed=SIM_SEED,
            )
        elif SIM_SETUP == "nonconstant":
            df = simulate_benchmark_nonconstant(
                n=N_SIM_NC,
                rho12=RHO12,
                rho13=RHO13,
                rho23=RHO23,
                beta_int=float(beta_int),
                eps_sd=EPS_SD_NC,
                seed=SIM_SEED,
            )
        else:
            raise ValueError("SIM_SETUP must be 'constant' or 'nonconstant'.")

        # ---- split ----
        real_train, real_hold = train_test_split(df, train_frac=0.7, seed=SPLIT_SEED)

        # ---- REAL baseline ----
        fit_real = smf.ols("V4 ~ V1 * V2 * V3", data=real_train).fit()
        beta123_real, se_real, p_real = extract_three_way_stats(fit_real, "V1:V2:V3")
        R2_real_train = float(fit_real.rsquared)

        pred_real_hold = fit_real.predict(real_hold).to_numpy()
        RMSE_real_hold = rmse(real_hold["V4"].to_numpy(), pred_real_hold)
        R2_real_hold = r2_oos(real_hold["V4"].to_numpy(), pred_real_hold)

        print(
            f"\n[BETA_INT={beta_int:.3f}] Setup={SIM_SETUP} | "
            f"REAL baseline: RMSE={RMSE_real_hold:.6f}, R2={R2_real_hold:.6f} | "
            f"beta123_real={beta123_real:.6f}"
        )

        out_dir_beta = os.path.join(OUT_ROOT, f"beta_int_{beta_int:.3f}")
        os.makedirs(out_dir_beta, exist_ok=True)

        for eps in eps_grid:
            eps_int = int(eps)
            if eps_int not in SIGMA_BY_EPS:
                raise ValueError(f"Missing SIGMA mapping for epsilon={eps_int}")
            if eps_int not in EPOCHS_BY_EPS:
                raise ValueError(f"Missing EPOCHS mapping for epsilon={eps_int}")

            sigma = float(SIGMA_BY_EPS[eps_int])
            epochs = int(EPOCHS_BY_EPS[eps_int])

            for r in reps:
                seed_run = 100_000 + 1_000 * eps_int + int(r) + 10_000_000 * b_idx

                synth_path = os.path.join(
                    out_dir_beta, f"eps_{eps_int}", f"synthetic_rep_{int(r):02d}.csv"
                )
                os.makedirs(os.path.dirname(synth_path), exist_ok=True)

                try:
                    Xsyn = dpwgan_synthesize(
                        real_train[cols],
                        seed=int(seed_run),
                        sigma=float(sigma),
                        epochs=int(epochs),
                        noise_dim=int(NOISE_DIM),
                        hidden_dim=int(HIDDEN_DIM),
                        n_critics=int(N_CRITICS),
                        learning_rate=float(LR),
                        weight_clip=float(WEIGHT_CLIP),
                        device=DEVICE,
                    )
                    Xsyn.to_csv(synth_path, index=False)
                except Exception as e:
                    rows.append(
                        {
                            "method": "DPWGAN",
                            "sim_setup": SIM_SETUP,
                            "beta_int": float(beta_int),
                            "epsilon": float(eps_int),
                            "rep": int(r),
                            "seed": int(seed_run),
                            "status": "FAIL",
                            "error": str(e),
                            "synth_file": synth_path,
                            "sigma": float(sigma),
                            "epochs": int(epochs),
                        }
                    )
                    print(f"[FAIL] beta_int={beta_int:.3f} eps={eps_int} rep={r}: {e}")
                    continue

                # ---- fit on synth ----
                fit_syn = smf.ols("V4 ~ V1 * V2 * V3", data=Xsyn).fit()
                beta123_syn, se_syn, p_syn = extract_three_way_stats(fit_syn, "V1:V2:V3")
                R2_syn_train = float(fit_syn.rsquared)

                # ---- eval on real holdout ----
                pred_hold = fit_syn.predict(real_hold).to_numpy()
                RMSE_hold = rmse(real_hold["V4"].to_numpy(), pred_hold)
                R2_hold = r2_oos(real_hold["V4"].to_numpy(), pred_hold)

                abs_err_beta123 = abs(beta123_real - beta123_syn)
                delta_RMSE = RMSE_hold - RMSE_real_hold
                delta_R2 = R2_hold - R2_real_hold

                rows.append(
                    {
                        "method": "DPWGAN",
                        "sim_setup": SIM_SETUP,
                        "beta_int": float(beta_int),
                        "epsilon": float(eps_int),
                        "rep": int(r),
                        "seed": int(seed_run),
                        "status": "OK",
                        "synth_file": synth_path,
                        "sigma": float(sigma),
                        "epochs": int(epochs),
                        "beta123_real": beta123_real,
                        "se_real": se_real,
                        "p_real": p_real,
                        "R2_real_train": R2_real_train,
                        "beta123_syn": beta123_syn,
                        "se_syn": se_syn,
                        "p_syn": p_syn,
                        "R2_syn_train": R2_syn_train,
                        "abs_err_beta123": abs_err_beta123,
                        "RMSE_holdout": RMSE_hold,
                        "R2_holdout": R2_hold,
                        "RMSE_real_holdout": RMSE_real_hold,
                        "R2_real_holdout": R2_real_hold,
                        "delta_RMSE_holdout": delta_RMSE,
                        "delta_R2_holdout": delta_R2,
                    }
                )

                print(
                    f"[BETA_INT={beta_int:.3f}] Done: eps={eps_int} rep={r} | "
                    f"abs_err={abs_err_beta123:.4f} | "
                    f"RMSE_hold={RMSE_hold:.4f} (Δ={delta_RMSE:.4f}) | "
                    f"R2_hold={R2_hold:.4f} (Δ={delta_R2:.4f})"
                )

    results_df = pd.DataFrame(rows)

    out_main = os.path.join(OUT_ROOT, f"higher_order_dpwgan_{SIM_SETUP}_betaIntSweep_results_final.csv")
    results_df.to_csv(out_main, index=False)

    ok = results_df[results_df["status"] == "OK"].copy()
    if not ok.empty:
        summary_df = (
            ok.groupby(["method", "sim_setup", "beta_int", "epsilon"], as_index=False)
            .agg(
                abs_err_mean=("abs_err_beta123", "mean"),
                abs_err_sd=("abs_err_beta123", "std"),
                rmse_mean=("RMSE_holdout", "mean"),
                rmse_sd=("RMSE_holdout", "std"),
                r2_mean=("R2_holdout", "mean"),
                r2_sd=("R2_holdout", "std"),
                delta_rmse_mean=("delta_RMSE_holdout", "mean"),
                delta_rmse_sd=("delta_RMSE_holdout", "std"),
                delta_r2_mean=("delta_R2_holdout", "mean"),
                delta_r2_sd=("delta_R2_holdout", "std"),
            )
        )
    else:
        summary_df = pd.DataFrame()

    out_sum = os.path.join(OUT_ROOT, f"higher_order_dpwgan_{SIM_SETUP}_betaIntSweep_summary_final.csv")
    summary_df.to_csv(out_sum, index=False)

    print("\nWrote:")
    print(f" - {out_main}")
    print(f" - {out_sum}")
    print(f" - outputs under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
