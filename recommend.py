# recommend.py
import torch
import torch.nn as nn
import joblib
import pandas as pd

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# load artifacts saved by your training step
scaler = joblib.load("scaler.pkl")
workout_encoder = joblib.load("workout_encoder.pkl")
meal_encoder = joblib.load("meal_encoder.pkl")

# instantiate models (dimensions must match what you trained)
input_dim = len(scaler.mean_)
workout_model = SimpleNN(input_dim, len(workout_encoder.classes_))
meal_model = SimpleNN(input_dim, len(meal_encoder.classes_))

workout_model.load_state_dict(torch.load("workout_model.pth"))
meal_model.load_state_dict(torch.load("meal_model.pth"))

workout_model.eval()
meal_model.eval()

def postprocess_meal(meal_label, row_dict):
    suggestions = []
    if row_dict.get("protein_g", 0) < 60:
        suggestions.append("Increase protein intake")
    if row_dict.get("carbs_g", 0) > 300:
        suggestions.append("Reduce carbs")
    if row_dict.get("fat_g", 0) > 90:
        suggestions.append("Cut down fats")
    if row_dict.get("sleep_hours", 0) < 7:
        suggestions.append("Aim for more sleep")
    if suggestions:
        return meal_label + " — " + ", ".join(suggestions)
    return meal_label + " ✅"

def recommend(row_dict):
    # row_dict must have the exact numeric features your scaler expects, in the same order
    df = pd.DataFrame([row_dict])
    X = scaler.transform(df.values)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        w_logits = workout_model(X_tensor)
        m_logits = meal_model(X_tensor)
    w_label = workout_encoder.inverse_transform([int(torch.argmax(w_logits, dim=1).item())])[0]
    m_label = meal_encoder.inverse_transform([int(torch.argmax(m_logits, dim=1).item())])[0]
    m_label = postprocess_meal(m_label, row_dict)
    return {"workout": w_label, "meal": m_label}
