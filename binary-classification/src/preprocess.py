import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)

    # Encode labels
    df["class"] = df["class"].map({"Normal": 0, "Abnormal": 1})

    X = df.drop(columns=["class"]).values
    y = df["class"].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 🔥 SAVE THE SCALER
    joblib.dump(scaler, "../artifacts/scaler.pkl")


    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test
