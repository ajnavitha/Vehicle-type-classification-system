import numpy as np
from sklearn.svm import SVC
import joblib

print("🔥 Creating model.pkl...")

# SMALL DATA (fast)
X = np.random.rand(50, 5)
y = np.random.randint(0, 5, 50)

# SIMPLE MODEL (fast training)
model = SVC(kernel='linear')

model.fit(X, y)

# SAVE
joblib.dump(model, "model.pkl")

print("✅ model.pkl created successfully!")