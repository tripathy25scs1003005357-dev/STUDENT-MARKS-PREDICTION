import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load Excel Dataset
df = pd.read_excel("student_marks_prediction_100_responses.xlsx")

# Show data
print("First 5 Records:\n", df.head())

# Step 2: Define input (X) and output (y)
X = df[['Hours_Studied']]
y = df['Marks_Obtained']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Prediction on test data
y_pred = model.predict(X_test)

# Step 6: Accuracy check
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 7: User input prediction
hours = float(input("\nEnter study hours: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks: {predicted_marks[0]:.2f}")

# Step 8: Visualization
plt.figure(figsize=(7,5))
plt.scatter(df['Hours_Studied'], df['Marks_Obtained'], label="Actual Data")
plt.plot(df['Hours_Studied'], model.predict(df[['Hours_Studied']]), color='red', label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Student Marks Prediction (100 Responses)")
plt.legend()
plt.show()