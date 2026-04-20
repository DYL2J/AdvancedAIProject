import random

import numpy as np
import pandas as pd


# Configuration
NUM_USERS = 200
NUM_PRODUCTS = 150
MIN_ORDERS = 10
MAX_ORDERS = 25
MIN_PRODUCTS_PER_ORDER = 3
MAX_PRODUCTS_PER_ORDER = 8
NUM_STAPLE_PRODUCTS = 8
STAPLE_SELECTION_PROB = 0.6
MAX_LOYALTY_BONUS = 0.3
LOYALTY_BONUS_STEP = 0.1
MAX_REORDER_PROB = 0.95
RANDOM_SEED = 30

CATEGORIES = [
    "dairy",
    "fruit",
    "vegetables",
    "meat",
    "snacks",
    "drinks",
    "bakery",
    "frozen",
]

CATEGORY_REORDER_PROB = {
    "dairy": 0.75,
    "fruit": 0.6,
    "vegetables": 0.55,
    "meat": 0.5,
    "snacks": 0.65,
    "drinks": 0.7,
    "bakery": 0.6,
    "frozen": 0.45,
}


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def create_users(num_users: int) -> pd.DataFrame:
    """Create a DataFrame of synthetic users."""
    return pd.DataFrame({"user_id": range(1, num_users + 1)})


def create_products(num_products: int, categories: list[str]) -> pd.DataFrame:
    """Create a DataFrame of synthetic products with categories."""
    return pd.DataFrame(
        {
            "product_id": range(1, num_products + 1),
            "category": np.random.choice(categories, num_products),
        }
    )


def select_products_for_order(
    staple_products: set[int],
    all_product_ids: list[int],
    num_products: int,
) -> set[int]:
    """Select products for an order with bias toward staple products."""
    chosen_products = set()

    while len(chosen_products) < num_products:
        if random.random() < STAPLE_SELECTION_PROB:
            chosen_products.add(random.choice(list(staple_products)))
        else:
            chosen_products.add(random.choice(all_product_ids))

    return chosen_products


def calculate_reordered_flag(
    product_id: int,
    previous_count: int,
    product_category_map: dict[int, str],
) -> int:
    """Determine whether a product is marked as reordered."""
    if previous_count == 0:
        return 0

    category = product_category_map[product_id]
    base_prob = CATEGORY_REORDER_PROB[category]
    loyalty_bonus = min(LOYALTY_BONUS_STEP * previous_count, MAX_LOYALTY_BONUS)
    reorder_prob = min(base_prob + loyalty_bonus, MAX_REORDER_PROB)

    return np.random.binomial(1, reorder_prob)


def generate_orders_and_order_products(
    users: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic orders and ordered products data."""
    orders = []
    order_products = []
    order_id = 1

    all_product_ids = products["product_id"].tolist()
    product_category_map = dict(zip(products["product_id"], products["category"]))

    for user_id in users["user_id"]:
        num_orders = random.randint(MIN_ORDERS, MAX_ORDERS)
        staple_products = set(
            np.random.choice(
                all_product_ids,
                size=NUM_STAPLE_PRODUCTS,
                replace=False,
            )
        )
        user_product_history = {}

        for order_number in range(1, num_orders + 1):
            orders.append(
                {
                    "order_id": order_id,
                    "user_id": user_id,
                    "order_number": order_number,
                }
            )

            num_products = random.randint(
                MIN_PRODUCTS_PER_ORDER,
                MAX_PRODUCTS_PER_ORDER,
            )

            chosen_products = select_products_for_order(
                staple_products=staple_products,
                all_product_ids=all_product_ids,
                num_products=num_products,
            )

            for product_id in chosen_products:
                previous_count = user_product_history.get(product_id, 0)
                reordered = calculate_reordered_flag(
                    product_id=product_id,
                    previous_count=previous_count,
                    product_category_map=product_category_map,
                )

                order_products.append(
                    {
                        "order_id": order_id,
                        "product_id": product_id,
                        "reordered": reordered,
                    }
                )

                user_product_history[product_id] = previous_count + 1

            order_id += 1

    orders_df = pd.DataFrame(orders)
    order_products_df = pd.DataFrame(order_products)

    return orders_df, order_products_df


def save_dataframes_to_csv(
    users: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
    order_products: pd.DataFrame,
) -> None:
    """Save generated DataFrames to CSV files."""
    users.to_csv("users.csv", index=False)
    products.to_csv("products.csv", index=False)
    orders.to_csv("orders.csv", index=False)
    order_products.to_csv("order_products.csv", index=False)


def main() -> None:
    """Generate and save the synthetic grocery reorder dataset."""
    set_random_seeds(RANDOM_SEED)

    users = create_users(NUM_USERS)
    products = create_products(NUM_PRODUCTS, CATEGORIES)
    orders_df, order_products_df = generate_orders_and_order_products(
        users,
        products,
    )

    save_dataframes_to_csv(
        users=users,
        products=products,
        orders=orders_df,
        order_products=order_products_df,
    )

    print("Synthetic dataset generated successfully.")


if __name__ == "__main__":
    main()