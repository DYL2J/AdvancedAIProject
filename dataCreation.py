import numpy as np
import pandas as pd
import random

#setup
NUM_USERS = 200
NUM_PRODUCTS = 150
MIN_ORDERS = 10
MAX_ORDERS = 25
MIN_PRODUCTS_PER_ORDER = 3
MAX_PRODUCTS_PER_ORDER = 8

random.seed(42)
np.random.seed(42)

#users
users = pd.DataFrame({
    "user_id": range(1, NUM_USERS + 1)
})

#product generation with categories
categories = [
    "dairy", "fruit", "vegetables", "meat", "snacks",
    "drinks", "bakery", "frozen"
]

products = pd.DataFrame({
    "product_id": range(1, NUM_PRODUCTS + 1),
    "category": np.random.choice(categories, NUM_PRODUCTS)
})

# Category-level reorder tendency
category_reorder_prob = {
    "dairy": 0.75,
    "fruit": 0.6,
    "vegetables": 0.55,
    "meat": 0.5,
    "snacks": 0.65,
    "drinks": 0.7,
    "bakery": 0.6,
    "frozen": 0.45,
}


orders = []
order_products = []

order_id = 1

for user_id in users["user_id"]:
    num_orders = random.randint(MIN_ORDERS, MAX_ORDERS)

    # Each user has common items ordered more often
    staple_products = set(
        np.random.choice(products["product_id"], size=8, replace=False)
    )

    # Track user-product purchase counts
    user_product_history = {}

    for order_number in range(1, num_orders + 1):
        orders.append({
            "order_id": order_id,
            "user_id": user_id,
            "order_number": order_number
        })

        num_products = random.randint(
            MIN_PRODUCTS_PER_ORDER, MAX_PRODUCTS_PER_ORDER
        )

        # Bias towards staples but allow variety
        chosen_products = set()
        while len(chosen_products) < num_products:
            if random.random() < 0.6:
                chosen_products.add(random.choice(list(staple_products)))
            else:
                chosen_products.add(
                    random.choice(products["product_id"].tolist())
                )

        for product_id in chosen_products:
            previous_count = user_product_history.get(product_id, 0)

            # First time purchase cannot be a reorder
            if previous_count == 0:
                reordered = 0
            else:
                category = products.loc[
                    products["product_id"] == product_id, "category"
                ].values[0]

                base_prob = category_reorder_prob[category]
                loyalty_bonus = min(0.1 * previous_count, 0.3)
                reorder_prob = min(base_prob + loyalty_bonus, 0.95)

                reordered = np.random.binomial(1, reorder_prob)

            order_products.append({
                "order_id": order_id,
                "product_id": product_id,
                "reordered": reordered
            })

            user_product_history[product_id] = previous_count + 1

        order_id += 1


orders_df = pd.DataFrame(orders)
order_products_df = pd.DataFrame(order_products)

#csv
users.to_csv("users.csv", index=False)
products.to_csv("products.csv", index=False)
orders_df.to_csv("orders.csv", index=False)
order_products_df.to_csv("order_products.csv", index=False)

print("Synthetic dataset generated successfully.")
