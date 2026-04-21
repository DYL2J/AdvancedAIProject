"""Retrain the reorder recommendation model with three additional features.

New features added:
    - days_since_last_purchase: Recency signal.  Derived from the order's
      position within the user's order history (order_number).
    - organic_preference: Per-user fraction of purchases that were organic-
      certified products.  Synthetic organic labels are assigned to 25 % of
      products and the preference score computed per user.

Exports:
    reorder_model.joblib           
    product_purchase_counts.joblib 
    product_reorder_rates.joblib   
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths
BASE_DIR = pathlib.Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "Modelling" / "ProdPred" / "Dataset"

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# Load data
print("Loading CSV data…")
users = pd.read_csv(DATASET_DIR / "users.csv")
products = pd.read_csv(DATASET_DIR / "products.csv")
orders = pd.read_csv(DATASET_DIR / "orders.csv")
order_products = pd.read_csv(DATASET_DIR / "order_products.csv")

data = (
    order_products
    .merge(orders, on="order_id")
    .merge(products, on="product_id")
)

# Original features
user_order_counts = orders.groupby("user_id")["order_id"].count()
data["user_total_orders"] = data["user_id"].map(user_order_counts)

product_purchase_counts = data.groupby("product_id").size()
product_reorder_rates = data.groupby("product_id")["reordered"].mean()

data["product_total_purchases"] = data["product_id"].map(product_purchase_counts)
data["product_reorder_rate"] = data["product_id"].map(product_reorder_rates)

user_product_counts = (
    data.groupby(["user_id", "product_id"])
    .size()
    .rename("user_product_purchase_count")
    .reset_index()
)
data = data.merge(user_product_counts, on=["user_id", "product_id"], how="left")



# New feature 1: days_since_last_purchase
max_order_per_user = orders.groupby("user_id")["order_number"].max()
data["max_order_number"] = data["user_id"].map(max_order_per_user)
data["order_recency_frac"] = (
    data["order_number"] / data["max_order_number"].clip(lower=1)
)
# Days since: newest purchase → 0 days, oldest → up to 365 days
data["days_since_last_purchase"] = (
    (1.0 - data["order_recency_frac"]) * 365
).astype(int)


unique_pids = products["product_id"].unique()


# New feature 2: organic_preference
organic_pids = set(
    rng.choice(unique_pids, size=int(len(unique_pids) * 0.25), replace=False)
)
products["is_organic"] = products["product_id"].isin(organic_pids).astype(int)
data = data.merge(
    products[["product_id", "is_organic"]],
    on="product_id",
    how="left",
    suffixes=("", "_dup"),
)

user_organic_pref = (
    data.groupby("user_id")["is_organic"].mean().rename("organic_preference")
)
data["organic_preference"] = data["user_id"].map(user_organic_pref)


# Feature lists
FEATURES = [
    "user_total_orders",
    "product_total_purchases",
    "product_reorder_rate",
    "user_product_purchase_count",
    "days_since_last_purchase",
    "organic_preference",
    "category",
]

NUMERIC_FEATURES = [
    "user_total_orders",
    "product_total_purchases",
    "product_reorder_rate",
    "user_product_purchase_count",
    "days_since_last_purchase",
    "organic_preference",
]

CATEGORICAL_FEATURES = ["category"]

X = data[FEATURES]
y = data["reordered"]


# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    ]
)


# Train and evaluate
print("Splitting data…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Training logistic regression with new features…")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n--- Evaluation ---")
print(classification_report(y_test, y_pred, digits=3))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")


# Export
local_path = BASE_DIR / "reorder_model.joblib"
joblib.dump(pipeline, local_path)
print(f"\nModel saved to {local_path}")

print(f"To deploy: copy {local_path} to recommender/artifacts/reorder_model.joblib")

joblib.dump(product_purchase_counts, BASE_DIR / "product_purchase_counts.joblib")
joblib.dump(product_reorder_rates, BASE_DIR / "product_reorder_rates.joblib")

print("\nDone. Model artifacts saved.")
