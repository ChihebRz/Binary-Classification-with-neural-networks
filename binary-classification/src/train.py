import torch
import torch.nn as nn
from preprocess import load_data
from model import SpineClassifier

# Load data
X_train, X_test, y_train, y_test = load_data("../data/column_2C_weka.csv")

# Model
model = SpineClassifier()

# Loss & optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 2000

for epoch in range(epochs):
    model.train()

    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
torch.save(model.state_dict(), "../artifacts/model.pth")

