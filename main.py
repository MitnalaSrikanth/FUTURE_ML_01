import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================
# LOAD DATASETS
# =====================================

print("Loading datasets...")

train = pd.read_csv("train.csv")
features = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")

print("Train Shape :", train.shape)
print("Features Shape :", features.shape)
print("Stores Shape :", stores.shape)

# =====================================
# MERGE DATASETS
# =====================================

df = pd.merge(train, features, on=["Store", "Date"], how="left")
df = pd.merge(df, stores, on="Store", how="left")

# =====================================
# DATA CLEANING
# =====================================

df.fillna(0, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])

# Date Features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Convert Store Type
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

print("\nDataset after merge:", df.shape)

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop(["Weekly_Sales", "Date"], axis=1)
y = df["Weekly_Sales"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================
# MODEL TRAINING
# =====================================

print("\nTraining model...")

model = RandomForestRegressor(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# =====================================
# PREDICTIONS
# =====================================

y_pred = model.predict(X_test)

# =====================================
# MODEL EVALUATION
# =====================================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("MAE      :", round(mae, 2))
print("R2 Score :", round(r2, 4))

# =====================================
# FEATURE IMPORTANCE GRAPH
# =====================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(12,6))
plt.bar(
    importance["Feature"][:10],
    importance["Importance"][:10]
)
plt.xticks(rotation=45)
plt.title("Top 10 Important Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# =====================================
# MONTHLY SALES TREND
# =====================================

monthly_sales = df.groupby("Month")["Weekly_Sales"].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(kind="bar")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# =====================================
# STORE WISE SALES
# =====================================

store_sales = df.groupby("Store")["Weekly_Sales"].sum()

plt.figure(figsize=(12,6))
store_sales.plot(kind="bar")
plt.title("Store Wise Sales")
plt.xlabel("Store")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# =====================================
# HOLIDAY IMPACT
# =====================================

holiday_sales = df.groupby("IsHoliday")["Weekly_Sales"].mean()

plt.figure(figsize=(6,5))
holiday_sales.plot(kind="bar")
plt.title("Average Sales: Holiday vs Non-Holiday")
plt.xlabel("Is Holiday")
plt.ylabel("Average Sales")
plt.grid(True)
plt.show()

# =====================================
# TEMPERATURE VS SALES
# =====================================

plt.figure(figsize=(10,6))
plt.scatter(
    df["Temperature"],
    df["Weekly_Sales"],
    alpha=0.3
)

plt.title("Temperature vs Weekly Sales")
plt.xlabel("Temperature")
plt.ylabel("Weekly Sales")
plt.show()

# =====================================
# UNEMPLOYMENT VS SALES
# =====================================

plt.figure(figsize=(10,6))
plt.scatter(
    df["Unemployment"],
    df["Weekly_Sales"],
    alpha=0.3
)

plt.title("Unemployment vs Weekly Sales")
plt.xlabel("Unemployment")
plt.ylabel("Weekly Sales")
plt.show()

# =====================================
# ACTUAL VS PREDICTED SALES
# =====================================

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values[:200],
    label="Actual Sales"
)

plt.plot(
    y_pred[:200],
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Records")
plt.ylabel("Weekly Sales")
plt.legend()
plt.grid(True)

plt.show()

# =====================================
# SALES DISTRIBUTION
# =====================================

plt.figure(figsize=(10,6))

plt.hist(
    df["Weekly_Sales"],
    bins=50
)

plt.title("Sales Distribution")
plt.xlabel("Weekly Sales")
plt.ylabel("Frequency")

plt.show()

# =====================================
# SAVE RESULTS
# =====================================

results = pd.DataFrame({
    "Actual_Sales": y_test.values,
    "Predicted_Sales": y_pred
})

results.to_csv(
    "forecast_results.csv",
    index=False
)

print("\nforecast_results.csv saved successfully")
print("\nProject Completed Successfully!")