import joblib
import pandas as pd


MIN_USER_ID = 1
MAX_USER_ID = 200
DEFAULT_TOP_N = 5

FEATURES = [
    "user_total_orders",
    "product_total_purchases",
    "product_reorder_rate",
    "user_product_purchase_count",
    "category",
]


def load_artifacts() -> tuple:
    """Load model artifacts and input datasets."""
    model = joblib.load("reorder_model.joblib")
    product_purchase_counts = joblib.load("product_purchase_counts.joblib")
    product_reorder_rates = joblib.load("product_reorder_rates.joblib")
    user_product_counts_df = pd.read_csv("user_product_purchase_counts.csv")
    products = pd.read_csv("products.csv")
    orders = pd.read_csv("orders.csv")

    return (
        model,
        product_purchase_counts,
        product_reorder_rates,
        user_product_counts_df,
        products,
        orders,
    )


def validate_user_id(user_id: int) -> None:
    """Validate that the user ID is within the expected range."""
    if not MIN_USER_ID <= user_id <= MAX_USER_ID:
        raise ValueError(
            f"User ID must be between {MIN_USER_ID} and {MAX_USER_ID}."
        )


def build_user_feature_frame(
    user_id: int,
    products: pd.DataFrame,
    orders: pd.DataFrame,
    user_product_counts_df: pd.DataFrame,
    product_purchase_counts: dict,
    product_reorder_rates: dict,
) -> pd.DataFrame:
    """Build the feature DataFrame for a single user across all products."""
    user_data = products.copy()
    user_data["user_id"] = user_id

    user_total_orders = orders[orders["user_id"] == user_id].shape[0]
    user_data["user_total_orders"] = user_total_orders

    user_history = user_product_counts_df[
        user_product_counts_df["user_id"] == user_id
    ][["product_id", "user_product_purchase_count"]]

    user_data = user_data.merge(user_history, on="product_id", how="left")

    user_data["user_product_purchase_count"] = user_data[
        "user_product_purchase_count"
    ].fillna(0)

    user_data["product_total_purchases"] = user_data["product_id"].map(
        product_purchase_counts
    )
    user_data["product_reorder_rate"] = user_data["product_id"].map(
        product_reorder_rates
    )

    return user_data


def recommend_products_for_user(
    user_id: int,
    model,
    products: pd.DataFrame,
    orders: pd.DataFrame,
    user_product_counts_df: pd.DataFrame,
    product_purchase_counts: dict,
    product_reorder_rates: dict,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Generate top-N reorder recommendations for a given user."""
    validate_user_id(user_id)

    user_data = build_user_feature_frame(
        user_id=user_id,
        products=products,
        orders=orders,
        user_product_counts_df=user_product_counts_df,
        product_purchase_counts=product_purchase_counts,
        product_reorder_rates=product_reorder_rates,
    )

    x_user = user_data[FEATURES]
    user_data["reorder_probability"] = model.predict_proba(x_user)[:, 1]

    recommendations = user_data.sort_values(
        by="reorder_probability",
        ascending=False,
    ).head(top_n)

    return recommendations[["product_id", "category", "reorder_probability"]]


def print_recommendations(recommendations: pd.DataFrame) -> None:
    """Print recommended products in a readable format."""
    print(f"\nTop {len(recommendations)} recommended products:\n")

    for _, row in recommendations.iterrows():
        product_id = int(row["product_id"])
        category = row["category"]
        probability = row["reorder_probability"]

        print(
            f"Product {product_id} ({category}) -> "
            f"Probability: {probability:.3f}"
        )


def main() -> None:
    """Run the grocery reorder recommendation system."""
    print("Grocery Reorder Recommendation System")

    try:
        (
            model,
            product_purchase_counts,
            product_reorder_rates,
            user_product_counts_df,
            products,
            orders,
        ) = load_artifacts()

        user_id = int(input(f"Enter user ID ({MIN_USER_ID}-{MAX_USER_ID}): "))

        recommendations = recommend_products_for_user(
            user_id=user_id,
            model=model,
            products=products,
            orders=orders,
            user_product_counts_df=user_product_counts_df,
            product_purchase_counts=product_purchase_counts,
            product_reorder_rates=product_reorder_rates,
        )

        print_recommendations(recommendations)

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()