import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOAD DATA
# ============================================================

ORDERS_PATH = "orders.csv"
ORDER_PRODUCTS_PATH = "order_products.csv"
PRODUCTS_PATH = "products.csv"

orders = pd.read_csv(ORDERS_PATH)
order_products = pd.read_csv(ORDER_PRODUCTS_PATH)
products = pd.read_csv(PRODUCTS_PATH)

print("Loaded data:")
print(f"orders: {orders.shape}")
print(f"order_products: {order_products.shape}")
print(f"products: {products.shape}")


# ============================================================
# 2. BUILD TRAINING DATASET
# ============================================================
# We create features using only information available BEFORE
# the current purchase row, to avoid leakage.

def build_dataset(orders_df, order_products_df, products_df):
    # Merge all relevant tables
    df = (
        order_products_df
        .merge(orders_df, on="order_id", how="left")
        .merge(products_df, on="product_id", how="left")
    )

    # Sort chronologically per user
    df = df.sort_values(
        by=["user_id", "order_number", "order_id", "product_id"]
    ).reset_index(drop=True)

    # Basic historical features
    df["user_total_prior_orders"] = df["order_number"] - 1

    # Number of items user has purchased before this row
    df["user_total_prior_items"] = df.groupby("user_id").cumcount()

    # Number of times THIS user has bought THIS product before
    df["user_product_prior_purchases"] = df.groupby(
        ["user_id", "product_id"]
    ).cumcount()

    # Number of times THIS product has been purchased globally before
    df["product_prior_purchases_global"] = df.groupby("product_id").cumcount()

    # Number of distinct products the user had bought before this row
    user_seen_products = {}
    distinct_prior = []

    for user_id, product_id in zip(df["user_id"].to_numpy(), df["product_id"].to_numpy()):
        seen = user_seen_products.setdefault(user_id, set())
        distinct_prior.append(len(seen))
        seen.add(product_id)

    df["user_distinct_products_prior"] = distinct_prior

    # Keep original merged dataframe for reference
    return df


df = build_dataset(orders, order_products, products)

print("\nMerged training dataframe shape:", df.shape)
print(df.head())


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

FEATURE_COLUMNS_NUMERIC = [
    "user_id",
    "product_id",
    "user_total_prior_orders",
    "user_total_prior_items",
    "user_product_prior_purchases",
    "product_prior_purchases_global",
    "user_distinct_products_prior"
]

TARGET_COLUMN = "reordered"

# One-hot encode category
X = df[FEATURE_COLUMNS_NUMERIC + ["category"]].copy()
X = pd.get_dummies(X, columns=["category"], drop_first=False)

y = df[TARGET_COLUMN].astype(int)

feature_columns_final = X.columns.tolist()

print("\nNumber of final features:", len(feature_columns_final))


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================
# A realistic split: use each user's LAST order as test data,
# and all earlier orders as training data.

max_order_per_user = df.groupby("user_id")["order_number"].transform("max")
test_mask = df["order_number"] == max_order_per_user
train_mask = ~test_mask

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

test_df = df.loc[test_mask].copy()

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train positive rate:", y_train.mean())
print("Test positive rate:", y_test.mean())


# ============================================================
# 5. TRAIN XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel training complete.")


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n================ MODEL EVALUATION ================")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))


# ============================================================
# 7. BUILD RECOMMENDATION DATA FOR TOP-5 REORDERS
# ============================================================
# We now build a "current state" for each user-product pair using
# all available historical purchases, then score likely reorders.

def build_recommendation_frame(full_df, products_df, feature_columns):
    # How many times each user bought each product in total
    user_product_state = (
        full_df.groupby(["user_id", "product_id"], as_index=False)
        .agg(user_product_prior_purchases=("product_id", "size"))
    )

    # User-level totals
    user_state = (
        full_df.groupby("user_id", as_index=False)
        .agg(
            user_total_prior_items=("product_id", "size"),
            user_distinct_products_prior=("product_id", "nunique"),
            user_total_prior_orders=("order_number", "max")
        )
    )

    # Global product popularity
    product_state = (
        full_df.groupby("product_id", as_index=False)
        .agg(product_prior_purchases_global=("product_id", "size"))
    )

    rec_df = (
        user_product_state
        .merge(user_state, on="user_id", how="left")
        .merge(product_state, on="product_id", how="left")
        .merge(products_df, on="product_id", how="left")
    )

    # Reorder only really makes sense for products already bought before,
    # so we can keep only products user has purchased at least once.
    rec_df = rec_df[rec_df["user_product_prior_purchases"] >= 1].copy()

    # Build feature matrix in exactly same format as training
    rec_X = rec_df[
        [
            "user_id",
            "product_id",
            "user_total_prior_orders",
            "user_total_prior_items",
            "user_product_prior_purchases",
            "product_prior_purchases_global",
            "user_distinct_products_prior",
            "category"
        ]
    ].copy()

    rec_X = pd.get_dummies(rec_X, columns=["category"], drop_first=False)

    # Match training columns exactly
    rec_X = rec_X.reindex(columns=feature_columns, fill_value=0)

    return rec_df, rec_X


recommendation_df, recommendation_X = build_recommendation_frame(
    full_df=df,
    products_df=products,
    feature_columns=feature_columns_final
)

recommendation_df["reorder_probability"] = model.predict_proba(recommendation_X)[:, 1]


# ============================================================
# 8. FUNCTION: TOP 5 PRODUCTS A USER IS MOST LIKELY TO REORDER
# ============================================================

def get_top_5_reorders(user_id, recommendation_scores_df):
    user_rows = recommendation_scores_df[
        recommendation_scores_df["user_id"] == user_id
    ].copy()

    if user_rows.empty:
        print(f"\nNo historical purchases found for user_id={user_id}")
        return pd.DataFrame()

    top5 = user_rows.sort_values(
        by="reorder_probability",
        ascending=False
    ).head(5)

    return top5[
        [
            "user_id",
            "product_id",
            "category",
            "user_product_prior_purchases",
            "product_prior_purchases_global",
            "reorder_probability"
        ]
    ]


# ============================================================
# 9. EXAMPLE USAGE
# ============================================================

example_user_id = 1
top5_for_user = get_top_5_reorders(example_user_id, recommendation_df)

print(f"\n================ TOP 5 PREDICTED REORDERS FOR USER {example_user_id} ================")
print(top5_for_user.to_string(index=False))


# ============================================================
# 10. OPTIONAL: INTERACTIVE USER INPUT
# ============================================================

while True:
    user_input = input("\nEnter a user_id to get top 5 reorder predictions (or 'q' to quit): ").strip()

    if user_input.lower() == "q":
        print("Exiting.")
        break

    if not user_input.isdigit():
        print("Please enter a valid numeric user_id.")
        continue

    chosen_user_id = int(user_input)
    result = get_top_5_reorders(chosen_user_id, recommendation_df)

    if result.empty:
        print("No results found for that user.")
    else:
        print(result.to_string(index=False))