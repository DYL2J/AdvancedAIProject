import joblib
import pandas as pd

#load models and artefeacts
model = joblib.load("reorder_model.joblib")
product_purchase_counts = joblib.load("product_purchase_counts.joblib")
product_reorder_rates = joblib.load("product_reorder_rates.joblib")
user_product_counts_df = pd.read_csv(
    "user_product_purchase_counts.csv"
)


products = pd.read_csv("products.csv")
orders = pd.read_csv("orders.csv")

FEATURES = [
    "user_total_orders",
    "product_total_purchases",
    "product_reorder_rate",
    "user_product_purchase_count",
    "category"
]

#logic for generating recommendations
def recommend_products_for_user(user_id, top_n=5):
    if user_id < 1 or user_id > 200:
        raise ValueError("User ID must be between 1 and 200")

    user_data = products.copy()
    user_data["user_id"] = user_id

    # User-level feature
    user_total_orders = orders[orders["user_id"] == user_id].shape[0]
    user_data["user_total_orders"] = user_total_orders

    # Correct user–product purchase counts
    user_history = user_product_counts_df[
        user_product_counts_df["user_id"] == user_id
    ][["product_id", "user_product_purchase_count"]]

    user_data = user_data.merge(
        user_history,
        on="product_id",
        how="left"
    )

    user_data["user_product_purchase_count"] = (
        user_data["user_product_purchase_count"].fillna(0)
    )

    # Product-level features
    user_data["product_total_purchases"] = user_data["product_id"].map(
        product_purchase_counts
    )

    user_data["product_reorder_rate"] = user_data["product_id"].map(
        product_reorder_rates
    )

    X_user = user_data[FEATURES]

    user_data["reorder_probability"] = model.predict_proba(X_user)[:, 1]

    recommendations = user_data.sort_values(
        "reorder_probability",
        ascending=False
    ).head(top_n)

    return recommendations[
        ["product_id", "category", "reorder_probability"]
    ]



#output
if __name__ == "__main__":
    print("Grocery Reorder Recommendation System")

    try:
        user_id = int(input("Enter user ID (1–200): "))
        results = recommend_products_for_user(user_id)

        print("\nTop 5 recommended products:\n")
        for _, row in results.iterrows():
            print(
                f"Product {int(row['product_id'])} "
                f"({row['category']}) → "
                f"Probability: {row['reorder_probability']:.3f}"
            )

    except Exception as e:
        print(f"Error: {e}")
