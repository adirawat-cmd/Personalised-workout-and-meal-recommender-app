import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# ================== Load Data ==================
df = pd.read_csv("synthetic_fitbit_labeled.csv")

# Features = only numeric columns (drop IDs and labels)
X = df.drop(columns=["user_id", "date", "workout_label", "nutrition_label"]).values
y_workout = df["workout_label"].values
y_meal = df["nutrition_label"].values

# ================== Encode Targets ==================
workout_encoder = LabelEncoder()
meal_encoder = LabelEncoder()

y_workout = workout_encoder.fit_transform(y_workout)
y_meal = meal_encoder.fit_transform(y_meal)

# ================== Scale Features ==================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ================== Train-Test Split ==================
X_train, X_test, yw_train, yw_test, ym_train, ym_test = train_test_split(
    X, y_workout, y_meal, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
yw_train = torch.tensor(yw_train, dtype=torch.long)
yw_test = torch.tensor(yw_test, dtype=torch.long)
ym_train = torch.tensor(ym_train, dtype=torch.long)
ym_test = torch.tensor(ym_test, dtype=torch.long)

# ================== Define Neural Net ==================
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

# Create models
workout_model = SimpleNN(X.shape[1], len(workout_encoder.classes_))
meal_model = SimpleNN(X.shape[1], len(meal_encoder.classes_))

# Loss & Optimizers
criterion = nn.CrossEntropyLoss()
optimizer_w = optim.Adam(workout_model.parameters(), lr=0.001)
optimizer_m = optim.Adam(meal_model.parameters(), lr=0.001)

# ================== Training Function ==================
def train_model(model, optimizer, X_train, y_train, epochs=50):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

# Train both models
workout_model = train_model(workout_model, optimizer_w, X_train, yw_train)
meal_model = train_model(meal_model, optimizer_m, X_train, ym_train)

# ================== Evaluation ==================
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).numpy()
    return accuracy_score(y_test.numpy(), preds)

acc_w = evaluate_model(workout_model, X_test, yw_test)
acc_m = evaluate_model(meal_model, X_test, ym_test)

print(f"✅ Workout Model Accuracy: {acc_w:.2f}")
print(f"✅ Meal Model Accuracy: {acc_m:.2f}")

# ================== Save Models & Encoders ==================
torch.save(workout_model.state_dict(), "workout_model.pth")
torch.save(meal_model.state_dict(), "meal_model.pth")
joblib.dump(workout_encoder, "workout_encoder.pkl")
joblib.dump(meal_encoder, "meal_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("🎉 Models trained, evaluated, and saved!")
