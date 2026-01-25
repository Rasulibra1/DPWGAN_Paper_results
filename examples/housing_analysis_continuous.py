# scripts/run_gan_minimal.py
# DPWGAN (continuous numeric) runner + SynthEval evaluation
# - loops epsilons x seeds
# - batch_size computed from sample_rate (ceil(q*n))
# - trains continuous DPWGAN (no one-hot) for numeric housing data
# - handles binary target column stored as "high"/"low" by mapping to 0/1 for training
# - maps synthetic target back to "high"/"low" for evaluation
# - fixes SynthEval crash on pandas StringDtype by casting to object
# - writes tidy metrics CSV + raw flattened CSV + epsilon summary CSV

from pathlib import Path
import random
import logging
import math

import pandas as pd
import numpy as np
import torch

from dpwgan import DPWGAN  # your dpwgan.py class


# -----------------------
# CONFIG
# -----------------------
ROOT = Path(__file__).resolve().parents[1]

REAL_CSV    = ROOT / "data" / "housing_train_final.csv"
HOLDOUT_CSV = ROOT / "data" / "housing_holdout_final.csv"

EVAL_DIR    = ROOT / "eval" / "metrics_raw" / "housing" / "dpwgan"
SUMMARY_DIR = ROOT / "eval" / "summaries"

EPS_LIST   = [1, 2, 3, 4, 5]
SEED_RANGE = range(10, 16)

# GAN hyperparams
NOISE_DIM   = 20
HIDDEN_DIM  = 20
N_CRITICS   = 5
LR          = 1e-4
WEIGHT_CLIP = 1 / HIDDEN_DIM

# Your epsilon -> epochs mapping (noise multiplier fixed at 2)
EPOCHS_BY_EPS = {
    1: 20,
    2: 74,
    3: 143,
    4: 244,
    5: 383,
}

# DP params (for training in this dpwgan.py: only sigma + weight_clip are used)
NOISE_MULTIPLIER = 2.0
SAMPLE_RATE = 0.01

# Target column (categorical labels: "high"/"low")
TARGET_COLUMN = "median_house_value_binary"
TARGET_LOW_LABEL = "low"
TARGET_HIGH_LABEL = "high"


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
    Fix SynthEval crash on pandas StringDtype by converting to object.
    """
    df = df.copy()
    string_cols = df.select_dtypes(include=["string"]).columns
    for c in string_cols:
        df[c] = df[c].astype("object")
    return df


def align_to_real_dtypes(real: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort cast 'other' columns to match 'real' dtypes.
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


def create_continuous_gan(noise_dim: int, hidden_dim: int, data_dim: int) -> DPWGAN:
    """
    Continuous DPWGAN:
      - generator outputs data_dim floats in [-1, 1] via tanh
      - discriminator scores data vectors
    """
    generator = torch.nn.Sequential(
        torch.nn.Linear(noise_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, data_dim),
        torch.nn.Tanh(),
    )

    discriminator = torch.nn.Sequential(
        torch.nn.Linear(data_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )

    def noise_fn(n: int):
        return torch.randn(n, noise_dim)

    return DPWGAN(generator, discriminator, noise_fn)


def scale_to_m1_1(df: pd.DataFrame):
    """
    Columnwise min-max scale to [-1, 1].
    Returns scaled_df(float32), mins, maxs.
    """
    mins = df.min(axis=0)
    maxs = df.max(axis=0)
    denom = (maxs - mins).replace(0, 1.0)
    scaled_0_1 = (df - mins) / denom
    scaled_m1_1 = 2.0 * scaled_0_1 - 1.0
    return scaled_m1_1.astype(np.float32), mins, maxs


def inverse_scale_from_m1_1(arr: np.ndarray, mins: pd.Series, maxs: pd.Series):
    """
    Inverse of columnwise [-1, 1] scaling.
    """
    arr_0_1 = (arr + 1.0) / 2.0
    return arr_0_1 * (maxs.values - mins.values) + mins.values


def map_target_to_01(series: pd.Series) -> pd.Series:
    """
    Map 'low'/'high' (case-insensitive) to 0/1.
    """
    s = series.astype(str).str.strip().str.lower()
    if not set(s.unique()).issubset({TARGET_LOW_LABEL, TARGET_HIGH_LABEL}):
        raise ValueError(
            f"{TARGET_COLUMN} must contain only '{TARGET_LOW_LABEL}'/'{TARGET_HIGH_LABEL}' "
            f"(case-insensitive). Found: {sorted(set(s.unique()))}"
        )
    return (s == TARGET_HIGH_LABEL).astype(np.int64)


def map_01_to_target_labels(series01: pd.Series) -> pd.Series:
    """
    Map 0/1 to 'low'/'high' strings.
    """
    series01 = series01.astype(int)
    return series01.map({0: TARGET_LOW_LABEL, 1: TARGET_HIGH_LABEL}).astype("object")


def safe_gan_train(gan, *, data, epochs, batch_size):
    """
    Your dpwgan.py train() supports: sigma + weight_clip (no delta/sample_rate args).
    """
    return gan.train(
        data=data,
        epochs=epochs,
        n_critics=N_CRITICS,
        batch_size=batch_size,
        learning_rate=LR,
        weight_clip=WEIGHT_CLIP,
        sigma=NOISE_MULTIPLIER,
    )


def main():
    logging.basicConfig(level=logging.INFO)

    # 1) Load train + optional holdout
    df_raw = pd.read_csv(REAL_CSV)

    if TARGET_COLUMN not in df_raw.columns:
        raise ValueError(f"TARGET_COLUMN='{TARGET_COLUMN}' not found in {REAL_CSV}")

    if HOLDOUT_CSV.exists():
        holdout_raw = pd.read_csv(HOLDOUT_CSV)
        # align column order
        holdout_raw = holdout_raw[df_raw.columns.intersection(holdout_raw.columns)]
        holdout_raw = holdout_raw.reindex(columns=df_raw.columns)
    else:
        holdout_raw = None

    # Keep evaluation copies with labels (high/low)
    df_eval_real = df_raw.copy()
    holdout_eval_real = holdout_raw.copy() if holdout_raw is not None else None

    # 2) Build numeric training DataFrame: map target to 0/1
    df_train = df_raw.copy()
    df_train[TARGET_COLUMN] = map_target_to_01(df_train[TARGET_COLUMN])

    if holdout_raw is not None:
        holdout_train = holdout_raw.copy()
        holdout_train[TARGET_COLUMN] = map_target_to_01(holdout_train[TARGET_COLUMN])
    else:
        holdout_train = None

    # Ensure all other columns are numeric (raise if not)
    for c in df_train.columns:
        if c == TARGET_COLUMN:
            continue
        df_train[c] = pd.to_numeric(df_train[c], errors="raise")
        if holdout_train is not None:
            holdout_train[c] = pd.to_numeric(holdout_train[c], errors="raise")

    # Setup output dirs
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    from syntheval import SynthEval

    all_metrics = []

    raw_csv = EVAL_DIR / "final_housing_dpwgan_raw_full_eval_all_eps_seeds_present_cont.csv"
    raw_csv_exists = raw_csv.exists()

    n = len(df_train)

    # 3) Loop eps × seeds
    for epsilon in EPS_LIST:
        if epsilon not in EPOCHS_BY_EPS:
            raise ValueError(f"Missing epochs mapping for epsilon={epsilon}")
        epochs = EPOCHS_BY_EPS[epsilon]

        for seed in SEED_RANGE:
            set_all_seeds(seed)

            batch_size = batch_size_from_sample_rate(n, SAMPLE_RATE)
            print(
                f"\n=== DPWGAN run (epsilon {epsilon}, epochs {epochs}, seed {seed}) ===\n"
                f"n={n}, sample_rate={SAMPLE_RATE} -> batch_size={batch_size}"
            )

            # 4) Scale training data to [-1,1]
            scaled_df, mins, maxs = scale_to_m1_1(df_train)
            data_tensor = torch.tensor(scaled_df.values, dtype=torch.float32)

            # 5) Train continuous GAN
            gan = create_continuous_gan(NOISE_DIM, HIDDEN_DIM, data_dim=data_tensor.shape[1])
            safe_gan_train(gan, data=data_tensor, epochs=epochs, batch_size=batch_size)

            # 6) Generate synthetic rows and inverse-scale
            synth_scaled = gan.generate(len(df_train)).detach().cpu().numpy()
            synth_arr = inverse_scale_from_m1_1(synth_scaled, mins, maxs)
            synth_num = pd.DataFrame(synth_arr, columns=df_train.columns)

            # 7) Postprocess target: threshold -> 0/1, then map back to "low"/"high"
            synth_num[TARGET_COLUMN] = (synth_num[TARGET_COLUMN] >= 0.5).astype(int)
            synth_df = synth_num.copy()
            synth_df[TARGET_COLUMN] = map_01_to_target_labels(synth_df[TARGET_COLUMN])

            # 8) Prepare SynthEval inputs
            df_eval = make_syntheval_safe(df_eval_real)
            synth_eval = make_syntheval_safe(synth_df)
            holdout_eval = make_syntheval_safe(holdout_eval_real) if holdout_eval_real is not None else None

            # Cast synth/holdout dtypes to match real best-effort
            synth_eval = align_to_real_dtypes(df_eval, synth_eval)
            if holdout_eval is not None:
                holdout_eval = align_to_real_dtypes(df_eval, holdout_eval)

            S = SynthEval(df_eval, holdout_dataframe=holdout_eval) if holdout_eval is not None else SynthEval(df_eval)

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
                raw_df.insert(0, "sample_rate", SAMPLE_RATE)

                raw_df.to_csv(
                    raw_csv,
                    mode="a",
                    header=not raw_csv_exists,
                    index=False,
                )
                raw_csv_exists = True

    # 9) Combined tidy metrics CSV
    combined = pd.concat(all_metrics, ignore_index=True)
    combined_csv = EVAL_DIR / "final_housing_dpwgan_metrics_all_eps_seeds_10_31_cont.csv"
    combined.to_csv(combined_csv, index=False)
    print("✅ Combined metrics CSV:", combined_csv)

    # 10) mean/std summary per epsilon+metric
    combined["value_num"] = pd.to_numeric(combined["value"], errors="coerce")
    summary = (
        combined.groupby(["epsilon", "metric"])["value_num"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values(["epsilon", "metric"])
    )
    summary_csv = SUMMARY_DIR / "final_housing_dpwgan_metrics_summary_by_epsilon_cont.csv"
    summary.to_csv(summary_csv, index=False)
    print("✅ Summary CSV          :", summary_csv)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
