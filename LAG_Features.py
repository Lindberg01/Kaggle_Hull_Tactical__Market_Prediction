import pandas as pd

# --- Load your CSV ---
df = pd.read_csv("train.csv")

# --- Parameters ---
LAG_PERIODS = [1, 3, 5]
ROLL_WINDOW = 5
features_to_use = ["E19", "V7", "V10", "P5", "S8"]

# --- Function to add lagged and rolling features ---
def add_lags_and_rolling(df: pd.DataFrame, features: list) -> pd.DataFrame:
    df = df.copy()
    for col in features:
        if col not in df.columns:
            continue
        # Lagged features
        for lag in LAG_PERIODS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag).fillna(0)
        # Rolling mean
        df[f"{col}_rollmean{ROLL_WINDOW}"] = df[col].rolling(window=ROLL_WINDOW).mean().fillna(0)
    return df

# --- Apply function ---
df_updated = add_lags_and_rolling(df, features_to_use)

# --- Save updated CSV ---
df_updated.to_csv("train_with_lags.csv", index=False)
print("Saved CSV with lagged and rolling features: 'train_with_lags.csv'")
