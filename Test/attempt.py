import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

# =============================
# 1. Load your data
# =============================
train_df = pd.read_csv("train_with_lags_2.csv", low_memory=False)
test_df = pd.read_csv("test_with_lags_2.csv", low_memory=False)


# Try to convert everything that looks like a number into a numeric dtype
train_df = train_df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
test_df  = test_df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

target_col = "lagged_forward_returns"  # ✅ Correct name
# Remove leading/trailing spaces if any
test_df['V7'] = test_df['V7'].astype(str).str.strip()
train_df['V7'] = train_df['V7'].astype(str).str.strip()

# Convert to numeric; invalid entries become NaN
train_df['V7'] = pd.to_numeric(train_df['V7'], errors='coerce')
test_df['V7'] = pd.to_numeric(test_df['V7'], errors='coerce')
X = train_df.drop(columns=[target_col])
y = train_df[target_col]
test_df = test_df[X.columns]

# 2. Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define model
model = lgb.LGBMRegressor(
    objective="huber",
    alpha=0.9,         # delta-like parameter (between 0 and 1); 0.9 works for slightly outlier-prone targets
    boosting_type="gbdt",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(stopping_rounds=10000000000000000),
        lgb.log_evaluation(period=100)  # prints every 100 rounds
    ]
)

# 5. Predict
preds = model.predict(test_df)

# 6. Evaluate
y_pred_valid = model.predict(X_valid)

mse = mean_squared_error(y_valid, y_pred_valid)
rmse = np.sqrt(mse)
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation RMSE: {rmse:.4f}")

# 7. Save predictions
pd.DataFrame({"prediction": preds}).to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")