# scripts/run_gan_minimal.py
# DPWGAN (categorical) runner + SynthEval evaluation
# - loops epsilons x seeds
# - batch_size computed from sample_rate (ceil(q*n))
# - uses repo DPWGAN training loop (sigma + weight clipping) instead of create_categorical_gan
# - fixes SynthEval crash on pandas StringDtype by casting to object
# - writes tidy metrics CSV + raw flattened CSV + epsilon summary CSV

from pathlib import Path
import random
import logging
import math

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from dpwgan import CategoricalDataset
from dpwgan.dpwgan import DPWGAN  # <-- benchmark-style DPWGAN


# -----------------------
# CONFIG
# -----------------------
ROOT = Path(__file__).resolve().parents[1]

SAVE_SYNTH = True
SYNTH_DIR  = ROOT / "data" / "synth" / "wine" / "dpwgan"  # pick whatever path you like


REAL_CSV    = ROOT / "data" / "wine_train_final.csv"
HOLDOUT_CSV = ROOT / "data" / "wine_holdout_final.csv"

EVAL_DIR    = ROOT / "eval" / "metrics_raw" / "wine" / "dpwgan"
SUMMARY_DIR = ROOT / "eval" / "summaries"

EPS_LIST   = [1, 2, 3, 4, 5]
SEED_RANGE = range(10, 16)

# GAN hyperparams (match your minimal script)
NOISE_DIM   = 20
HIDDEN_DIM  = 20
N_CRITICS   = 5
LR          = 1e-4
WEIGHT_CLIP = 1 / HIDDEN_DIM  # keep your choice

# Your epsilon -> epochs mapping (sigma fixed at 2)
EPOCHS_BY_EPS = {
    1: 20,
    2: 74,
    3: 143,
    4: 244,
    5: 383,
}

# DP-ish params you provided
# Note: DPWGAN repo loop typically uses sigma (+ clipping) but may not consume delta/sample_rate.
NOISE_MULTIPLIER = 2.0
DELTA = 1e-5
SAMPLE_RATE = 0.01

TARGET_COLUMN = "type"


# -----------------------
# HELPERS
# -----------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def batch_size_from_sample_rate(n: int, sample_rate: float) -> int:
    bs = int(math.ceil(sample_rate * n))
    return max(1, min(n, bs))


def make_syntheval_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix SynthEval crash:
      TypeError: Cannot interpret '<StringDtype(...)>' as a data type
    by converting pandas StringDtype -> object.
    """
    df = df.copy()
    string_cols = df.select_dtypes(include=["string"]).columns
    for c in string_cols:
        df[c] = df[c].astype("object")
    return df


def align_to_real_dtypes(real: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort cast 'other' columns to match 'real' dtypes
    so SynthEval sees consistent types across real/synth/holdout.
    """
    other = other.copy()
    for c in real.columns:
        if c not in other.columns:
            continue
        rd = real[c].dtype
        if str(rd) == "string":
            rd = "object"
        try:
            other[c] = other[c].astype(rd)
        except Exception:
            pass
    return other


# -------------------------------
# Benchmark-style nets for one-hot flat vectors
# -------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),  # keep outputs in [0,1] so argmax-per-category behaves sensibly
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
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


def train_and_generate_dpwgan_onehot(
    data_tensor: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    sigma: float,
    noise_dim: int,
    hidden_dim: int,
    n_critics: int,
    learning_rate: float,
    weight_clip: float,
    device: str,
) -> np.ndarray:
    """
    Train DPWGAN on one-hot-flat data (treated as continuous in [0,1]) and return generated one-hot-flat array.
    Uses the repo DPWGAN training loop (benchmark-style): sigma + weight clipping.
    """
    set_all_seeds(seed)

    d = int(data_tensor.shape[1])
    G = Generator(noise_dim=noise_dim, out_dim=d, hidden_dim=hidden_dim).to(device)
    D = Discriminator(in_dim=d, hidden_dim=hidden_dim).to(device)

    noise_fn = lambda n: torch.randn(int(n), int(noise_dim), device=device)
    gan = DPWGAN(generator=G, discriminator=D, noise_function=noise_fn)

    gan.train(
        data=data_tensor,
        epochs=int(epochs),
        n_critics=int(n_critics),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        sigma=float(sigma) if sigma is not None else None,
        weight_clip=float(weight_clip),
    )

    with torch.no_grad():
        flat = gan.generate(int(data_tensor.shape[0])).detach().cpu().numpy()
    return flat


def main():
    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(REAL_CSV)

    if HOLDOUT_CSV.exists():
        holdout = pd.read_csv(HOLDOUT_CSV)
        holdout = holdout[df.columns.intersection(holdout.columns)]
        holdout = holdout.reindex(columns=df.columns)
    else:
        holdout = None

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    from syntheval import SynthEval

    all_metrics = []

    raw_csv = EVAL_DIR / "final_wine_dpwgan_raw_full_eval_all_eps_seeds_present.csv"
    raw_csv_exists = raw_csv.exists()

    n = len(df)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Using device=%s", device)

    for epsilon in EPS_LIST:
        if epsilon not in EPOCHS_BY_EPS:
            raise ValueError(f"Missing epochs mapping for epsilon={epsilon}")
        epochs = EPOCHS_BY_EPS[epsilon]

        for seed in SEED_RANGE:
            set_all_seeds(seed)

            batch_size = batch_size_from_sample_rate(n, SAMPLE_RATE)
            print(
                f"\n=== DPWGAN run (epsilon {epsilon}, epochs {epochs}, seed {seed}) ===\n"
                f"n={n}, sample_rate={SAMPLE_RATE} -> batch_size={batch_size} | sigma={NOISE_MULTIPLIER}"
            )

            # 1) Encode REAL train as one-hot flat
            dataset = CategoricalDataset(df)
            data_np = dataset.to_onehot_flat()
            data_tensor = torch.tensor(data_np, dtype=torch.float32, device=device)

            # 2) Train/generate using benchmark-style DPWGAN loop
            flat_synth = train_and_generate_dpwgan_onehot(
                data_tensor,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                sigma=NOISE_MULTIPLIER,
                noise_dim=NOISE_DIM,
                hidden_dim=HIDDEN_DIM,
                n_critics=N_CRITICS,
                learning_rate=LR,
                weight_clip=WEIGHT_CLIP,
                device=device,
            )

            # 3) Decode back to categorical dataframe
            synth_df = dataset.from_onehot_flat(flat_synth)

            # 3b) Save synthetic dataset (optional)
            if SAVE_SYNTH:
                SYNTH_DIR.mkdir(parents=True, exist_ok=True)
                synth_path = SYNTH_DIR / f"wine_dpwgan_eps_{int(epsilon)}_seed_{int(seed)}.csv"
                synth_df.to_csv(synth_path, index=False)


          

            # 4) SynthEval (same as your current script)
            df_eval = make_syntheval_safe(df)
            synth_eval = make_syntheval_safe(synth_df)
            holdout_eval = make_syntheval_safe(holdout) if holdout is not None else None

            synth_eval = align_to_real_dtypes(df_eval, synth_eval)
            if holdout_eval is not None:
                holdout_eval = align_to_real_dtypes(df_eval, holdout_eval)

            S = (
                SynthEval(df_eval, holdout_dataframe=holdout_eval)
                if holdout_eval is not None
                else SynthEval(df_eval)
            )

            try:
                eval_out = S.evaluate(synth_eval, TARGET_COLUMN, presets_file="full_eval")
            except Exception as e:
                print("SynthEval preset 'full_eval' not found; using default set.", e)
                eval_out = S.evaluate(synth_eval)

            # --------- A) tidy metrics ----------
            if isinstance(eval_out, dict):
                metrics_df = (
                    pd.DataFrame([eval_out])
                    .T.reset_index()
                    .rename(columns={"index": "metric", 0: "value"})
                )
            else:
                metrics_df = pd.DataFrame(eval_out)

            metrics_df.insert(0, "seed", seed)
            metrics_df.insert(0, "epsilon", epsilon)
            metrics_df.insert(0, "epochs", epochs)
            metrics_df.insert(0, "batch_size", batch_size)
            all_metrics.append(metrics_df)

            # --------- B) raw flattened results appended ----------
            raw = getattr(S, "_raw_results", None)
            if raw is not None:
                raw_df = pd.json_normalize(raw, sep=".")
                raw_df.insert(0, "seed", seed)
                raw_df.insert(0, "epsilon", epsilon)
                raw_df.insert(0, "epochs", epochs)
                raw_df.insert(0, "batch_size", batch_size)
                raw_df.insert(0, "noise_multiplier", NOISE_MULTIPLIER)
                raw_df.insert(0, "delta", DELTA)
                raw_df.insert(0, "sample_rate", SAMPLE_RATE)

                raw_df.to_csv(
                    raw_csv,
                    mode="a",
                    header=not raw_csv_exists,
                    index=False,
                )
                raw_csv_exists = True

    # 5) Combined tidy metrics CSV
    combined = pd.concat(all_metrics, ignore_index=True)
    combined_csv = EVAL_DIR / "final_wine_dpwgan_metrics_all_eps_seeds_10_31.csv"
    combined.to_csv(combined_csv, index=False)
    print("✅ Combined metrics CSV:", combined_csv)

    # 6) mean/std summary per epsilon+metric
    combined["value_num"] = pd.to_numeric(combined["value"], errors="coerce")
    summary = (
        combined.groupby(["epsilon", "metric"])["value_num"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values(["epsilon", "metric"])
    )
    summary_csv = SUMMARY_DIR / "final_wine_dpwgan_metrics_summary_by_epsilon.csv"
    summary.to_csv(summary_csv, index=False)
    print("✅ Summary CSV          :", summary_csv)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
