import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset
df = pd.read_csv("StudentPerformance.csv")

# Example: Suppose dataset has columns "Hours_Studied" and "Exam_Score"
X = df.iloc[:,:-1]   # Feature(s)
y = df.columns[-1]        # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "predictor/student_model.pkl")
print("Model trained and saved successfully!")
