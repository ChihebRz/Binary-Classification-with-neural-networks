import torch
import numpy as np
import joblib
from model import SpineClassifier

# Load scaler
scaler = joblib.load("../artifacts/scaler.pkl")



# Load model
model = SpineClassifier()
model.load_state_dict(torch.load("../artifacts/model.pth"))
model.eval()


def predict(sample: np.ndarray):
    """
    sample: np.array with shape (6,) or (1, 6)
    """

    if sample.ndim == 1:
        sample = sample.reshape(1, -1)

    # Scale input
    sample_scaled = scaler.transform(sample)

    # Convert to tensor
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    # Predict
    with torch.inference_mode():
        logit = model(sample_tensor)
        prob = torch.sigmoid(logit)
        pred = torch.round(prob)

    label = "Abnormal" if pred.item() == 1 else "Normal"

    return {
        "class": label,
        "probability": float(prob.item())
    }


# Test
if __name__ == "__main__":
    example = np.array([51.0, 23.4, 45.1, 27.6, 117.8, 12.3])
    print(predict(example))
