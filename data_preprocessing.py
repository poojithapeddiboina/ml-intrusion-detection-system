import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------
# 1️⃣ Load Data
# ---------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)

    # Remove extra spaces in column names
    df.columns = df.columns.str.strip()

    return df


# ---------------------------------------------------
# 2️⃣ Clean Data
# ---------------------------------------------------
def clean_data(df):

    # Remove infinite and missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Handle both possible label names
    if "Label" in df.columns:
        label_col = "Label"
    elif "Flow Label" in df.columns:
        label_col = "Flow Label"
    else:
        raise ValueError("Label column not found in dataset")

    # Create AttackType column
    df["AttackType"] = df[label_col]

    # Binary label: 0 = BENIGN, 1 = Attack
    df["BinaryLabel"] = df[label_col].apply(
        lambda x: 0 if str(x).upper() == "BENIGN" else 1
    )

    # Keep only numeric features
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Add labels back
    df_numeric["AttackType"] = df["AttackType"].values
    df_numeric["BinaryLabel"] = df["BinaryLabel"].values

    return df_numeric


# ---------------------------------------------------
# 3️⃣ Zero-Day Split
# ---------------------------------------------------
def split_data_zero_day(df, zero_day_attack):

    normal_data = df[df["AttackType"] == "BENIGN"]
    zero_day_data = df[df["AttackType"] == zero_day_attack]

    if len(zero_day_data) == 0:
        raise ValueError(f"{zero_day_attack} not found in dataset")

    # Train only on normal traffic
    X_train = normal_data.drop(["AttackType", "BinaryLabel"], axis=1)
    y_train = normal_data["BinaryLabel"]

    # Test = some normal + zero-day attack
    test_data = pd.concat([
        normal_data.sample(frac=0.3, random_state=42),
        zero_day_data
    ])

    X_test = test_data.drop(["AttackType", "BinaryLabel"], axis=1)
    y_test = test_data["BinaryLabel"]

    return X_train, X_train, X_test, y_train, y_test


# ---------------------------------------------------
# 4️⃣ Scaling
# ---------------------------------------------------
def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
