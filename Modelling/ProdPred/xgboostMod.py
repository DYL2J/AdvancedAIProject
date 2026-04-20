import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


ORDERS_PATH = "Dataset/orders.csv"
ORDER_PRODUCTS_PATH = "Dataset/order_products.csv"
PRODUCTS_PATH = "Dataset/products.csv"

TARGET_COLUMN = "reordered"

FEATURE_COLUMNS_NUMERIC = [
    "user_id",
    "product_id",
    "user_total_prior_orders",
    "user_total_prior_items",
    "user_product_prior_purchases",
    "product_prior_purchases_global",
    "user_distinct_products_prior",
]

CATEGORICAL_COLUMNS = ["category"]

MODEL_PARAMS = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

EXAMPLE_USER_ID = 1
TOP_N_REORDERS = 5


def load_data(
    orders_path: str,
    order_products_path: str,
    products_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load source CSV files."""
    orders_df = pd.read_csv(orders_path)
    order_products_df = pd.read_csv(order_products_path)
    products_df = pd.read_csv(products_path)

    print("Loaded data:")
    print(f"orders: {orders_df.shape}")
    print(f"order_products: {order_products_df.shape}")
    print(f"products: {products_df.shape}")

    return orders_df, order_products_df, products_df


def build_dataset(
    orders_df: pd.DataFrame,
    order_products_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the modelling dataset using only information available
    before the current purchase row to avoid leakage.
    """
    df = (
        order_products_df
        .merge(orders_df, on="order_id", how="left")
        .merge(products_df, on="product_id", how="left")
    )

    df = df.sort_values(
        by=["user_id", "order_number", "order_id", "product_id"]
    ).reset_index(drop=True)

    df["user_total_prior_orders"] = df["order_number"] - 1
    df["user_total_prior_items"] = df.groupby("user_id").cumcount()
    df["user_product_prior_purchases"] = df.groupby(
        ["user_id", "product_id"]
    ).cumcount()
    df["product_prior_purchases_global"] = df.groupby("product_id").cumcount()

    user_seen_products: dict[int, set[int]] = {}
    distinct_prior = []

    for user_id, product_id in zip(
        df["user_id"].to_numpy(),
        df["product_id"].to_numpy(),
    ):
        seen_products = user_seen_products.setdefault(user_id, set())
        distinct_prior.append(len(seen_products))
        seen_products.add(product_id)

    df["user_distinct_products_prior"] = distinct_prior

    return df


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare model feature matrix and target vector."""
    x_data = df[FEATURE_COLUMNS_NUMERIC + CATEGORICAL_COLUMNS].copy()
    x_data = pd.get_dummies(
        x_data,
        columns=CATEGORICAL_COLUMNS,
        drop_first=False,
    )

    y_data = df[TARGET_COLUMN].astype(int)
    feature_columns = x_data.columns.tolist()

    print(f"\nNumber of final features: {len(feature_columns)}")

    return x_data, y_data, feature_columns


def split_train_test_by_last_order(
    df: pd.DataFrame,
    x_data: pd.DataFrame,
    y_data: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Split data so that each user's last order is used for testing
    and all earlier orders are used for training.
    """
    max_order_per_user = df.groupby("user_id")["order_number"].transform("max")
    test_mask = df["order_number"] == max_order_per_user
    train_mask = ~test_mask

    x_train = x_data.loc[train_mask].copy()
    x_test = x_data.loc[test_mask].copy()
    y_train = y_data.loc[train_mask].copy()
    y_test = y_data.loc[test_mask].copy()
    test_df = df.loc[test_mask].copy()

    print(f"\nTrain shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape}")
    print(f"Train positive rate: {y_train.mean():.4f}")
    print(f"Test positive rate: {y_test.mean():.4f}")

    return x_train, x_test, y_train, y_test, test_df


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Train the XGBoost classifier."""
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(x_train, y_train)

    print("\nModel training complete.")
    return model


def evaluate_model(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Evaluate the trained model and print classification metrics."""
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n================ MODEL EVALUATION ================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))


def build_recommendation_frame(
    full_df: pd.DataFrame,
    products_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a recommendation dataset representing the current state of
    each user-product pair, then prepare it for scoring.
    """
    user_product_state = (
        full_df.groupby(["user_id", "product_id"], as_index=False)
        .agg(user_product_prior_purchases=("product_id", "size"))
    )

    user_state = (
        full_df.groupby("user_id", as_index=False)
        .agg(
            user_total_prior_items=("product_id", "size"),
            user_distinct_products_prior=("product_id", "nunique"),
            user_total_prior_orders=("order_number", "max"),
        )
    )

    product_state = (
        full_df.groupby("product_id", as_index=False)
        .agg(product_prior_purchases_global=("product_id", "size"))
    )

    recommendation_df = (
        user_product_state
        .merge(user_state, on="user_id", how="left")
        .merge(product_state, on="product_id", how="left")
        .merge(products_df, on="product_id", how="left")
    )

    recommendation_df = recommendation_df[
        recommendation_df["user_product_prior_purchases"] >= 1
    ].copy()

    recommendation_x = recommendation_df[
        [
            "user_id",
            "product_id",
            "user_total_prior_orders",
            "user_total_prior_items",
            "user_product_prior_purchases",
            "product_prior_purchases_global",
            "user_distinct_products_prior",
            "category",
        ]
    ].copy()

    recommendation_x = pd.get_dummies(
        recommendation_x,
        columns=["category"],
        drop_first=False,
    )

    recommendation_x = recommendation_x.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    return recommendation_df, recommendation_x


def add_recommendation_scores(
    model: XGBClassifier,
    recommendation_df: pd.DataFrame,
    recommendation_x: pd.DataFrame,
) -> pd.DataFrame:
    """Add reorder probability scores to the recommendation DataFrame."""
    scored_df = recommendation_df.copy()
    scored_df["reorder_probability"] = model.predict_proba(
        recommendation_x
    )[:, 1]
    return scored_df


def get_top_reorders_for_user(
    user_id: int,
    recommendation_scores_df: pd.DataFrame,
    top_n: int = TOP_N_REORDERS,
) -> pd.DataFrame:
    """Return the top-N predicted reorders for a given user."""
    user_rows = recommendation_scores_df[
        recommendation_scores_df["user_id"] == user_id
    ].copy()

    if user_rows.empty:
        print(f"\nNo historical purchases found for user_id={user_id}")
        return pd.DataFrame()

    top_reorders = user_rows.sort_values(
        by="reorder_probability",
        ascending=False,
    ).head(top_n)

    return top_reorders[
        [
            "user_id",
            "product_id",
            "category",
            "user_product_prior_purchases",
            "product_prior_purchases_global",
            "reorder_probability",
        ]
    ]


def run_interactive_prompt(recommendation_scores_df: pd.DataFrame) -> None:
    """Allow interactive lookup of top reorder predictions by user ID."""
    while True:
        user_input = input(
            "\nEnter a user_id to get top 5 reorder predictions "
            "(or 'q' to quit): "
        ).strip()

        if user_input.lower() == "q":
            print("Exiting.")
            break

        if not user_input.isdigit():
            print("Please enter a valid numeric user_id.")
            continue

        chosen_user_id = int(user_input)
        result = get_top_reorders_for_user(
            chosen_user_id,
            recommendation_scores_df,
        )

        if result.empty:
            print("No results found for that user.")
        else:
            print(result.to_string(index=False))


def main() -> None:
    """Run the full reorder prediction and recommendation pipeline."""
    orders_df, order_products_df, products_df = load_data(
        ORDERS_PATH,
        ORDER_PRODUCTS_PATH,
        PRODUCTS_PATH,
    )

    df = build_dataset(orders_df, order_products_df, products_df)

    print(f"\nMerged training dataframe shape: {df.shape}")
    print(df.head())

    x_data, y_data, feature_columns = prepare_features(df)

    x_train, x_test, y_train, y_test, _ = split_train_test_by_last_order(
        df,
        x_data,
        y_data,
    )

    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)

    recommendation_df, recommendation_x = build_recommendation_frame(
        full_df=df,
        products_df=products_df,
        feature_columns=feature_columns,
    )

    recommendation_scores_df = add_recommendation_scores(
        model,
        recommendation_df,
        recommendation_x,
    )

    top_reorders = get_top_reorders_for_user(
        EXAMPLE_USER_ID,
        recommendation_scores_df,
    )

    print(
        f"\n================ TOP 5 PREDICTED REORDERS FOR USER "
        f"{EXAMPLE_USER_ID} ================"
    )
    print(top_reorders.to_string(index=False))

    run_interactive_prompt(recommendation_scores_df)


if __name__ == "__main__":
    main()