import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("owid-covid-data.csv")

# Filter for one country
df = df[df["location"] == "India"]

# Select required columns
df = df[[
    "total_cases",
    "total_deaths",
    "total_vaccinations",
    "population",
    "new_cases"
]]

# Handle missing values
df = df.fillna(0)

# Features and target
X = df.drop("new_cases", axis=1)
y = df["new_cases"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Model Mean Absolute Error:", mae)

# Save model
joblib.dump(model, "covid_rf_model.pkl")
print("Model saved as covid_rf_model.pkl")
