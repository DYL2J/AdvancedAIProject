import random
import os
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

# Config
# ====================================================================================================
# config for the synthetic dataset (6 years of history, jan 2020 - dec 2025)
# three kinds of buyers sit on the platform:
#   - customers: individuals ordering irregularly (the `num_orders` random pool).
#   - restaurants: business buyers placing weekly recurring orders with a cuisine-driven
#     category bias (e.g. a steakhouse orders meat-heavy baskets).
#   - community_groups: schools / youth groups / charities doing 1-7 bulk orders per month.
num_customers = 750
num_orders = 7500                # random one-off customer orders, spread over the window
num_restaurants = 100             # each placing ~weekly recurring orders
num_community_groups = 125        # each placing 1-3 bulk orders / month (spaced)

# restaurant recurring-order cadence (see django RecurringOrder.FREQUENCY_CHOICES).
# ~4 orders/month ≈ weekly, with a small spread so not every restaurant is identical.
restaurant_orders_per_month_mean = 10.0
restaurant_orders_per_month_std  = 2.0

# community-group monthly bulk-order range (inclusive, uniform).
# kept low so consecutive bulks are spaced ~10-30 days apart, matching real
# wholesale / catering buying rhythms (community fridges, food banks, schools).
community_orders_per_month_min = 3
community_orders_per_month_max = 9

# Per Order Settings
min_products_per_order = 3
max_products_per_order = 10
num_staple_products = 8
staple_selection_prob = 0.6

# bulk orders (community groups) and restaurant recurring orders carry larger baskets
# than household customer orders - typical catering / wholesale patterns.
bulk_min_products_per_order = 6
bulk_max_products_per_order = 14
restaurant_min_products_per_order = 5
restaurant_max_products_per_order = 10

# reorder probability: category-specific base + loyalty bonus per prior purchase
loyalty_bonus_step = 0.05
max_loyalty_bonus = 0.25
max_reorder_prob = 0.95
random_seed = 300

# matches django seed.py COMMISSION_RATE
commission_rate = 0.05

# order / producer_order status mix (matches django Order.Status + ProducerOrder.Status)
# most historical orders are completed; a small tail reflects in-flight + cancelled
order_status_mix = [
    # (order_status, producer_order_status, weight)
    ("completed", "completed", 0.88),
    ("completed", "delivered", 0.04),
    ("cancelled", "cancelled", 0.04),
    ("confirmed", "ready",     0.015),
    ("confirmed", "preparing", 0.01),
    ("confirmed", "accepted",  0.01),
    ("pending",   "pending",   0.005),
]

# per-category base reorder probability (drives the ml target)
category_reorder_prob = {
    "dairy":      0.75,
    "fruit":      0.60,
    "vegetables": 0.55,
    "meat":       0.50,
    "snacks":     0.65,
    "drinks":     0.70,
    "bakery":     0.60,
    "frozen":     0.45,
}

base_date = datetime(2020, 1, 1)
end_date = datetime(2025, 12, 31)
order_hour_min = 8
order_hour_max = 22


def month_to_season(month: int) -> str:
    """Map month number (1-12) to meteorological season."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


ALL_SEASONS = {"spring", "summer", "autumn", "winter"}

# product-level seasonal availability.
# products not listed here are treated as all-season.
# note: we now use this as a SOFT penalty (see out_of_season_penalty below) rather
# than a hard mask - real supply chains still sell tomatoes in january, just at
# lower volumes from greenhouses / imports.
product_seasonality = {
    # fruit
    "strawberry": {"summer"},
    "blueberry": {"summer", "autumn"},
    "grapes": {"autumn"},
    "apple": {"autumn", "winter"},
    "mango": {"summer"},
    "pineapple": {"summer"},
    # vegetables
    "lettuce": {"spring", "summer"},
    "spinach": {"spring", "autumn", "winter"},
    "broccoli": {"autumn", "winter"},
    "kale": {"autumn", "winter"},
    "tomato": {"summer"},
    "cucumber": {"summer"},
    "bell pepper": {"summer", "autumn"},
    "mushroom": {"autumn", "winter", "spring"},
}

# out-of-season items still appear but with 25% of their in-season selection weight,
# reflecting reduced (but non-zero) availability via imports / glasshouse supply.
out_of_season_penalty = 0.25

# CATEGORY x SEASON soft demand weights.
# these bias which category is picked during product sampling so the MIX of an order
# shifts with the season rather than the AVAILABILITY flipping on/off.
# motivated by retail-seasonality evidence (uk ons retail sales seasonal adjustment factors,
# kantar worldpanel category insights, and maxoptra/nfff industry reports cited in report).
category_seasonality_weights = {
    "fruit":      {"spring": 1.2, "summer": 1.4, "autumn": 1.0, "winter": 0.7},
    "vegetables": {"spring": 1.1, "summer": 1.2, "autumn": 1.0, "winter": 0.9},
    "meat":       {"spring": 1.0, "summer": 0.9, "autumn": 1.2, "winter": 1.3},
    "snacks":     {"spring": 1.0, "summer": 1.1, "autumn": 1.0, "winter": 1.2},
    "drinks":     {"spring": 1.2, "summer": 1.5, "autumn": 0.9, "winter": 0.8},
    "bakery":     {"spring": 1.0, "summer": 0.9, "autumn": 1.1, "winter": 1.3},
    "frozen":     {"spring": 1.0, "summer": 1.1, "autumn": 1.0, "winter": 1.2},
    "dairy":      {"spring": 1.0, "summer": 0.95, "autumn": 1.1, "winter": 1.2},
}

# restaurant cuisines - category affinity weights.
# each cuisine has a realistic bias towards certain product categories. the bias
# is the main thing that makes restaurant reorder behaviour *learnable* vs customers.
# weights get normalised at sampling time.
restaurant_cuisines = {
    # name                    : (cuisine_label,      {category: affinity})
    "sunshine smoothies":       ("smoothie_shop",    {"fruit": 5, "dairy": 2, "drinks": 2, "snacks": 1}),
    "istanbul kebab house":     ("kebab",            {"meat": 5, "vegetables": 3, "bakery": 2, "dairy": 1}),
    "black angus steakhouse":   ("steakhouse",       {"meat": 6, "vegetables": 2, "dairy": 1, "bakery": 1}),
    "bella napoli":             ("italian",          {"vegetables": 3, "dairy": 3, "bakery": 2, "meat": 2}),
    "sweet layers bakery":      ("cake_shop",        {"dairy": 4, "bakery": 3, "snacks": 2, "fruit": 1}),
    "dough brothers pizzeria":  ("pizzeria",         {"dairy": 4, "vegetables": 2, "meat": 2, "bakery": 2}),
    "blue wave sushi":          ("sushi",            {"meat": 5, "vegetables": 3, "snacks": 1, "drinks": 1}),
    "green bowl cafe":          ("vegan_cafe",       {"vegetables": 4, "fruit": 3, "snacks": 2, "drinks": 1}),
    "morning brew bakery cafe": ("bakery_cafe",      {"bakery": 4, "dairy": 3, "drinks": 2, "fruit": 1}),
    "the fresh press":          ("juice_bar",        {"fruit": 5, "drinks": 3, "vegetables": 2, "dairy": 1}),
}

# community group templates - basket is broader (catering a group) and they order
# in monthly bursts rather than a weekly rhythm.
community_group_profiles = [
    "greenfield primary school",      "westbrook academy",
    "st margarets school",            "oakwood high school",
    "brookside youth group",          "riverside scouts",
    "hillview guides",                "st pauls church group",
    "trinity community kitchen",      "eastside food bank",
    "community fridge hub",           "age well club",
    "meadowbank nursery",             "sunshine kids club",
    "after school network",           "parents co-op",
    "woodlands charity",              "helping hands charity",
    "new horizons college",           "city youth centre",
    "harbourside senior club",        "sports academy juniors",
    "bright futures charity",         "welcome in refugee kitchen",
    "community dinner club",
]

# Data
# ====================================================================================================
# categories (django catalog.category: id, name, description)
categories_data = [
    {"id": 1, "name": "dairy", "description": "milk, cheese, yoghurt, butter and cream products"},
    {"id": 2, "name": "fruit", "description": "fresh and seasonal fruits"},
    {"id": 3, "name": "vegetables", "description": "fresh and seasonal vegetables"},
    {"id": 4, "name": "meat", "description": "fresh meat, poultry and seafood"},
    {"id": 5, "name": "snacks", "description": "crisps, nuts, bars and pantry staples"},
    {"id": 6, "name": "drinks", "description": "juices, soft drinks, tea and coffee"},
    {"id": 7, "name": "bakery", "description": "bread, pastries and baked goods"},
    {"id": 8, "name": "frozen", "description": "frozen meals, vegetables and desserts"},
]

category_name_to_id = {c["name"]: c["id"] for c in categories_data}

# producers and their products (maps to django producers.producer and catalog.product)
producers_raw = {
    "greenfarms": {
        "company_number": "GF-100001",
        "products": {
            "apple":       {"category": "fruit",      "price": 1.20, "unit": "kg"},
            "banana":      {"category": "fruit",      "price": 0.85, "unit": "kg"},
            "strawberry":  {"category": "fruit",      "price": 2.50, "unit": "kg"},
            "blueberry":   {"category": "fruit",      "price": 3.00, "unit": "kg"},
            "spinach":     {"category": "vegetables", "price": 1.50, "unit": "kg"},
            "broccoli":    {"category": "vegetables", "price": 1.30, "unit": "kg"},
            "carrot":      {"category": "vegetables", "price": 0.70, "unit": "kg"},
            "lettuce":     {"category": "vegetables", "price": 0.90, "unit": "unit"},
            "kale":        {"category": "vegetables", "price": 1.60, "unit": "kg"},
            "grapes":      {"category": "fruit",      "price": 2.20, "unit": "kg"},
        },
    },
    "freshdairy": {
        "company_number": "FD-100002",
        "products": {
            "whole milk":       {"category": "dairy", "price": 1.30, "unit": "litre"},
            "semi skimmed milk":{"category": "dairy", "price": 1.20, "unit": "litre"},
            "cheddar cheese":   {"category": "dairy", "price": 3.50, "unit": "kg"},
            "greek yoghurt":    {"category": "dairy", "price": 1.80, "unit": "unit"},
            "butter":           {"category": "dairy", "price": 2.10, "unit": "unit"},
            "cream cheese":     {"category": "dairy", "price": 1.60, "unit": "unit"},
            "double cream":     {"category": "dairy", "price": 1.40, "unit": "ml"},
            "cottage cheese":   {"category": "dairy", "price": 1.90, "unit": "unit"},
        },
    },
    "oceancatch": {
        "company_number": "OC-100003",
        "products": {
            "salmon fillet": {"category": "meat", "price": 6.50, "unit": "kg"},
            "cod fillet":    {"category": "meat", "price": 5.00, "unit": "kg"},
            "tuna steak":   {"category": "meat", "price": 7.00, "unit": "kg"},
            "prawns":        {"category": "meat", "price": 5.50, "unit": "kg"},
            "sea bass":     {"category": "meat", "price": 8.00, "unit": "kg"},
            "haddock":      {"category": "meat", "price": 4.80, "unit": "kg"},
            "mackerel":     {"category": "meat", "price": 3.90, "unit": "kg"},
        },
    },
    "meadowmeats": {
        "company_number": "MM-100004",
        "products": {
            "chicken breast": {"category": "meat", "price": 5.50, "unit": "kg"},
            "beef mince":     {"category": "meat", "price": 4.50, "unit": "kg"},
            "pork chops":     {"category": "meat", "price": 4.00, "unit": "kg"},
            "lamb leg":       {"category": "meat", "price": 9.00, "unit": "kg"},
            "bacon rashers":  {"category": "meat", "price": 3.20, "unit": "kg"},
            "sausages":       {"category": "meat", "price": 3.00, "unit": "kg"},
            "turkey mince":   {"category": "meat", "price": 4.80, "unit": "kg"},
            "steak":          {"category": "meat", "price": 10.00, "unit": "kg"},
        },
    },
    "golden bakehouse": {
        "company_number": "GB-100005",
        "products": {
            "sourdough loaf": {"category": "bakery", "price": 3.50, "unit": "unit"},
            "white bread":    {"category": "bakery", "price": 1.20, "unit": "unit"},
            "croissant":      {"category": "bakery", "price": 1.50, "unit": "unit"},
            "bagel":          {"category": "bakery", "price": 1.00, "unit": "unit"},
            "baguette":       {"category": "bakery", "price": 1.80, "unit": "unit"},
            "rye bread":      {"category": "bakery", "price": 2.50, "unit": "unit"},
            "cinnamon roll":  {"category": "bakery", "price": 2.00, "unit": "unit"},
            "scone":          {"category": "bakery", "price": 1.30, "unit": "unit"},
        },
    },
    "crunchtime": {
        "company_number": "CT-100006",
        "products": {
            "tortilla chips": {"category": "snacks", "price": 2.00, "unit": "unit"},
            "popcorn":        {"category": "snacks", "price": 1.50, "unit": "unit"},
            "trail mix":      {"category": "snacks", "price": 3.00, "unit": "unit"},
            "rice cakes":     {"category": "snacks", "price": 1.80, "unit": "unit"},
            "pretzels":       {"category": "snacks", "price": 1.60, "unit": "unit"},
            "granola bar":    {"category": "snacks", "price": 1.20, "unit": "unit"},
            "dark chocolate": {"category": "snacks", "price": 2.50, "unit": "unit"},
            "mixed nuts":     {"category": "snacks", "price": 3.50, "unit": "unit"},
            "fruit gummies":  {"category": "snacks", "price": 1.00, "unit": "unit"},
        },
    },
    "sipwell": {
        "company_number": "SW-100007",
        "products": {
            "orange juice":    {"category": "drinks", "price": 2.00, "unit": "litre"},
            "apple juice":     {"category": "drinks", "price": 1.80, "unit": "litre"},
            "sparkling water": {"category": "drinks", "price": 0.80, "unit": "litre"},
            "cola":            {"category": "drinks", "price": 1.20, "unit": "litre"},
            "lemonade":        {"category": "drinks", "price": 1.00, "unit": "litre"},
            "green tea":       {"category": "drinks", "price": 2.50, "unit": "unit"},
            "coffee beans":    {"category": "drinks", "price": 5.00, "unit": "kg"},
            "oat milk":        {"category": "drinks", "price": 1.60, "unit": "litre"},
        },
    },
    "frostbite foods": {
        "company_number": "FF-100008",
        "products": {
            "frozen pizza":   {"category": "frozen", "price": 3.00, "unit": "unit"},
            "fish fingers":   {"category": "frozen", "price": 2.50, "unit": "unit"},
            "frozen peas":    {"category": "frozen", "price": 1.20, "unit": "kg"},
            "ice cream tub":  {"category": "frozen", "price": 3.50, "unit": "unit"},
            "frozen chips":   {"category": "frozen", "price": 2.00, "unit": "kg"},
            "frozen berries": {"category": "frozen", "price": 2.80, "unit": "kg"},
            "frozen spinach": {"category": "frozen", "price": 1.50, "unit": "kg"},
        },
    },
    "harvest fields": {
        "company_number": "HF-100009",
        "products": {
            "potato":      {"category": "vegetables", "price": 0.80, "unit": "kg"},
            "onion":       {"category": "vegetables", "price": 0.60, "unit": "kg"},
            "tomato":      {"category": "vegetables", "price": 1.50, "unit": "kg"},
            "cucumber":    {"category": "vegetables", "price": 0.70, "unit": "unit"},
            "bell pepper": {"category": "vegetables", "price": 1.20, "unit": "unit"},
            "mushroom":    {"category": "vegetables", "price": 2.00, "unit": "kg"},
            "avocado":     {"category": "fruit",      "price": 1.50, "unit": "unit"},
            "mango":       {"category": "fruit",      "price": 1.80, "unit": "unit"},
            "pineapple":   {"category": "fruit",      "price": 2.00, "unit": "unit"},
        },
    },
    "pantry essentials": {
        "company_number": "PE-100010",
        "products": {
            "pasta":           {"category": "snacks", "price": 1.20, "unit": "kg"},
            "rice":            {"category": "snacks", "price": 1.50, "unit": "kg"},
            "olive oil":       {"category": "snacks", "price": 4.00, "unit": "litre"},
            "tinned tomatoes": {"category": "snacks", "price": 0.80, "unit": "unit"},
            "baked beans":     {"category": "snacks", "price": 0.70, "unit": "unit"},
            "peanut butter":   {"category": "snacks", "price": 2.50, "unit": "unit"},
            "honey":           {"category": "snacks", "price": 3.00, "unit": "unit"},
            "flour":           {"category": "bakery", "price": 1.00, "unit": "kg"},
            "eggs":            {"category": "dairy",  "price": 2.20, "unit": "dozen"},
        },
    },
    "orchard valley": {
        "company_number": "OV-100011",
        "products": {
            "pear":        {"category": "fruit", "price": 1.40, "unit": "kg"},
            "plum":        {"category": "fruit", "price": 1.70, "unit": "kg"},
            "raspberry":   {"category": "fruit", "price": 3.20, "unit": "kg"},
            "cherry":      {"category": "fruit", "price": 3.80, "unit": "kg"},
            "peach":       {"category": "fruit", "price": 2.40, "unit": "kg"},
            "melon":       {"category": "fruit", "price": 2.60, "unit": "unit"},
            "watermelon":  {"category": "fruit", "price": 3.50, "unit": "unit"},
        },
    },
    "alpine creamery": {
        "company_number": "AC-100012",
        "products": {
            "brie":             {"category": "dairy", "price": 3.80, "unit": "kg"},
            "mozzarella":       {"category": "dairy", "price": 2.60, "unit": "unit"},
            "parmesan":         {"category": "dairy", "price": 4.50, "unit": "kg"},
            "sour cream":       {"category": "dairy", "price": 1.70, "unit": "unit"},
            "mascarpone":       {"category": "dairy", "price": 2.40, "unit": "unit"},
            "feta":             {"category": "dairy", "price": 3.20, "unit": "kg"},
            "single cream":     {"category": "dairy", "price": 1.30, "unit": "ml"},
        },
    },
    "coastal catch": {
        "company_number": "CC-100013",
        "products": {
            "trout fillet":  {"category": "meat", "price": 6.20, "unit": "kg"},
            "sardines":      {"category": "meat", "price": 3.40, "unit": "kg"},
            "king prawns":   {"category": "meat", "price": 6.90, "unit": "kg"},
            "smoked salmon": {"category": "meat", "price": 9.50, "unit": "kg"},
            "crab meat":     {"category": "meat", "price": 8.20, "unit": "kg"},
        },
    },
    "heritage butchers": {
        "company_number": "HB-100014",
        "products": {
            "ribeye steak":   {"category": "meat", "price": 12.00, "unit": "kg"},
            "lamb mince":     {"category": "meat", "price": 6.20, "unit": "kg"},
            "duck breast":    {"category": "meat", "price": 8.50, "unit": "kg"},
            "pork belly":     {"category": "meat", "price": 5.40, "unit": "kg"},
            "chicken thighs": {"category": "meat", "price": 4.20, "unit": "kg"},
            "venison steak":  {"category": "meat", "price": 11.00, "unit": "kg"},
        },
    },
    "village bakery": {
        "company_number": "VB-100015",
        "products": {
            "wholemeal loaf":    {"category": "bakery", "price": 1.80, "unit": "unit"},
            "seeded loaf":       {"category": "bakery", "price": 2.20, "unit": "unit"},
            "pain au chocolat":  {"category": "bakery", "price": 1.80, "unit": "unit"},
            "brioche":           {"category": "bakery", "price": 2.60, "unit": "unit"},
            "ciabatta":          {"category": "bakery", "price": 1.90, "unit": "unit"},
            "flatbread":         {"category": "bakery", "price": 1.40, "unit": "unit"},
            "muffin":            {"category": "bakery", "price": 1.20, "unit": "unit"},
        },
    },
    "nutty nibbles": {
        "company_number": "NN-100016",
        "products": {
            "almonds":         {"category": "snacks", "price": 4.50, "unit": "kg"},
            "cashews":         {"category": "snacks", "price": 5.20, "unit": "kg"},
            "walnuts":         {"category": "snacks", "price": 5.00, "unit": "kg"},
            "hazelnut spread": {"category": "snacks", "price": 3.20, "unit": "unit"},
            "energy bar":      {"category": "snacks", "price": 1.40, "unit": "unit"},
            "protein bar":     {"category": "snacks", "price": 2.00, "unit": "unit"},
        },
    },
    "brewhouse co": {
        "company_number": "BC-100017",
        "products": {
            "ground coffee":  {"category": "drinks", "price": 6.00, "unit": "kg"},
            "earl grey tea":  {"category": "drinks", "price": 2.80, "unit": "unit"},
            "herbal tea":     {"category": "drinks", "price": 2.60, "unit": "unit"},
            "cold brew":      {"category": "drinks", "price": 2.80, "unit": "litre"},
            "almond milk":    {"category": "drinks", "price": 1.70, "unit": "litre"},
            "soy milk":       {"category": "drinks", "price": 1.60, "unit": "litre"},
        },
    },
    "arctic kitchen": {
        "company_number": "AK-100018",
        "products": {
            "frozen lasagne":      {"category": "frozen", "price": 3.80, "unit": "unit"},
            "frozen chicken pie":  {"category": "frozen", "price": 3.20, "unit": "unit"},
            "frozen prawns":       {"category": "frozen", "price": 5.80, "unit": "kg"},
            "frozen broccoli":     {"category": "frozen", "price": 1.40, "unit": "kg"},
            "frozen dumplings":    {"category": "frozen", "price": 3.00, "unit": "unit"},
            "sorbet tub":          {"category": "frozen", "price": 3.20, "unit": "unit"},
        },
    },
    "root and stem": {
        "company_number": "RS-100019",
        "products": {
            "courgette":   {"category": "vegetables", "price": 1.10, "unit": "kg"},
            "aubergine":   {"category": "vegetables", "price": 1.30, "unit": "kg"},
            "sweet potato": {"category": "vegetables", "price": 1.20, "unit": "kg"},
            "butternut squash": {"category": "vegetables", "price": 1.60, "unit": "unit"},
            "leek":        {"category": "vegetables", "price": 1.00, "unit": "kg"},
            "celery":      {"category": "vegetables", "price": 0.80, "unit": "unit"},
            "beetroot":    {"category": "vegetables", "price": 1.10, "unit": "kg"},
        },
    },
    "kitchen pantry co": {
        "company_number": "KP-100020",
        "products": {
            "lentils":           {"category": "snacks", "price": 1.80, "unit": "kg"},
            "chickpeas":         {"category": "snacks", "price": 1.00, "unit": "unit"},
            "coconut milk":      {"category": "snacks", "price": 1.20, "unit": "unit"},
            "soy sauce":         {"category": "snacks", "price": 2.20, "unit": "unit"},
            "sugar":             {"category": "bakery", "price": 1.00, "unit": "kg"},
            "baking powder":     {"category": "bakery", "price": 1.20, "unit": "unit"},
            "vanilla extract":   {"category": "bakery", "price": 2.80, "unit": "unit"},
            "cocoa powder":      {"category": "bakery", "price": 2.60, "unit": "unit"},
        },
    },
}

# Creating Datasets
# ====================================================================================================
# seed everything for reproducibility
def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# accounts table - customers + one account per producer (django accounts.account)
# django fields: id, username, email, first_name, last_name, account_type
_first_names = [
    "jane", "bob", "alice", "tom", "emma", "liam", "olivia", "noah", "ava", "ethan",
    "sophia", "mason", "isabella", "logan", "mia", "lucas", "amelia", "oliver", "harper", "elijah",
]
_last_names = [
    "doe", "smith", "jones", "brown", "taylor", "wilson", "davies", "evans", "thomas", "roberts",
    "walker", "wright", "hall", "green", "baker", "carter", "mitchell", "turner", "phillips", "parker",
]


def create_accounts(
    num_customers: int,
    producer_names: list[str],
    restaurant_names: list[str] | None = None,
    community_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build the accounts table covering all four account_types.

    Note: django Account.account_type supports customer / producer / restaurant /
    community_group / admin. We populate the first four; restaurants and community
    groups are buyer accounts in the same sense as customers but with distinct
    ordering behaviour driven by restaurant_cuisines and community_group_profiles.
    """
    restaurant_names = restaurant_names or []
    community_names  = community_names  or []

    rows = []
    aid = 1

    # customer accounts
    for i in range(1, num_customers + 1):
        rows.append({
            "id": aid,
            "username": f"customer{i}",
            "email": f"customer{i}@example.com",
            "first_name": random.choice(_first_names),
            "last_name": random.choice(_last_names),
            "account_type": "customer",
        })
        aid += 1

    # restaurant accounts (buyer-side business users)
    for rname in restaurant_names:
        slug = rname.replace(" ", "")
        rows.append({
            "id": aid,
            "username": slug,
            "email": f"{slug}@example.com",
            "first_name": "",
            "last_name": "",
            "account_type": "restaurant",
        })
        aid += 1

    # community-group accounts (schools, charities, youth groups, etc.)
    for gname in community_names:
        slug = gname.replace(" ", "")
        rows.append({
            "id": aid,
            "username": slug,
            "email": f"{slug}@example.com",
            "first_name": "",
            "last_name": "",
            "account_type": "community_group",
        })
        aid += 1

    # producer accounts (one per producer)
    for pname in producer_names:
        slug = pname.replace(" ", "")
        rows.append({
            "id": aid,
            "username": slug,
            "email": f"{slug}@example.com",
            "first_name": "",
            "last_name": "",
            "account_type": "producer",
        })
        aid += 1

    return pd.DataFrame(rows)


# addresses table - one delivery address per customer, one business address per producer
# django addresses.address: id, account_id, address_type, address_line_1, city, postcode, is_default
_uk_cities = [
    ("exeter", "EX"), ("bristol", "BS"), ("bath", "BA"), ("london", "SW"),
    ("manchester", "M"), ("leeds", "LS"), ("liverpool", "L"), ("birmingham", "B"),
    ("cardiff", "CF"), ("edinburgh", "EH"), ("glasgow", "G"), ("newcastle", "NE"),
    ("sheffield", "S"), ("nottingham", "NG"), ("oxford", "OX"), ("cambridge", "CB"),
]
_street_names = [
    "high street", "king road", "queen street", "church lane", "station road",
    "main street", "park avenue", "mill lane", "orchard road", "farm lane",
]


def _random_address(rng: random.Random) -> dict:
    city, prefix = rng.choice(_uk_cities)
    return {
        "address_line_1": f"{rng.randint(1, 250)} {rng.choice(_street_names)}",
        "city": city,
        "postcode": f"{prefix}{rng.randint(1, 20)} {rng.randint(1, 9)}{rng.choice('ABCDEFGH')}{rng.choice('ABCDEFGH')}",
    }


def create_addresses(accounts_df: pd.DataFrame) -> pd.DataFrame:
    """All buyer accounts (customer / restaurant / community_group) get a delivery
    address; producer accounts get a business address."""
    rng = random.Random(random_seed + 10)
    rows = []
    addr_id = 1
    buyer_types = {"customer", "restaurant", "community_group"}
    for _, acc in accounts_df.iterrows():
        addr = _random_address(rng)
        addr_type = "delivery" if acc["account_type"] in buyer_types else "business"
        rows.append({
            "id": addr_id,
            "account_id": int(acc["id"]),
            "address_type": addr_type,
            "address_line_1": addr["address_line_1"],
            "city": addr["city"],
            "postcode": addr["postcode"],
            "is_default": True,
        })
        addr_id += 1
    return pd.DataFrame(rows)


# customers table - one-to-one with every BUYER account.
# django's schema nests Organisation under Customer, so restaurants and community
# groups each need a Customer row too (to eventually host their Organisation record).
def create_customers(accounts_df: pd.DataFrame, addresses_df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(random_seed + 11)
    buyer_types = {"customer", "restaurant", "community_group"}
    buyer_accounts = accounts_df[accounts_df["account_type"].isin(buyer_types)]
    delivery_addr_by_account = dict(
        zip(addresses_df[addresses_df["address_type"] == "delivery"]["account_id"],
            addresses_df[addresses_df["address_type"] == "delivery"]["id"])
    )
    rows = []
    cid = 1
    for _, acc in buyer_accounts.iterrows():
        rows.append({
            "id": cid,
            "account_id": int(acc["id"]),
            "phone_number": f"07{rng.randint(100, 999)}{rng.randint(100000, 999999)}",
            "default_delivery_address_id": int(delivery_addr_by_account[acc["id"]]),
        })
        cid += 1
    return pd.DataFrame(rows)


# categories table
def create_categories() -> pd.DataFrame:
    return pd.DataFrame(categories_data)


# producers table (each linked to their producer account)
# django producers.producer: id, account_id, company_name, company_number,
#                            company_description, lead_time_hours, business_address_id
_producer_descriptions = {
    "greenfarms":        "family-run organic farm supplying fresh seasonal produce.",
    "freshdairy":        "award-winning dairy producer with artisan cheeses.",
    "oceancatch":        "sustainable fishery delivering daily catch.",
    "meadowmeats":       "ethically reared meat and poultry from local farms.",
    "golden bakehouse":  "traditional bakery specialising in sourdough and pastries.",
    "crunchtime":        "handmade snacks and trail mixes.",
    "sipwell":           "cold-pressed juices, teas and speciality drinks.",
    "frostbite foods":   "quality frozen meals and ready-to-cook ingredients.",
    "harvest fields":    "wholesale fruit and vegetable grower.",
    "pantry essentials": "everyday pantry staples and store cupboard basics.",
    "orchard valley":    "orchard-grown stone fruit and berries.",
    "alpine creamery":   "speciality european-style cheeses and cream.",
    "coastal catch":     "day-boat seafood from the south coast.",
    "heritage butchers": "premium meat cuts from heritage-breed livestock.",
    "village bakery":    "slow-fermented breads and continental pastries.",
    "nutty nibbles":     "roasted nuts, energy bars and pantry snacks.",
    "brewhouse co":      "speciality coffee, tea and plant milks.",
    "arctic kitchen":    "chef-quality frozen ready meals and ingredients.",
    "root and stem":     "heritage-variety root vegetables and squashes.",
    "kitchen pantry co": "store-cupboard staples, baking and world foods.",
}


def create_producers(
    accounts_df: pd.DataFrame,
    producer_names: list[str],
    addresses_df: pd.DataFrame,
) -> pd.DataFrame:
    rng = random.Random(random_seed + 12)
    producer_accounts = accounts_df[accounts_df["account_type"] == "producer"]
    business_addr_by_account = dict(
        zip(addresses_df[addresses_df["address_type"] == "business"]["account_id"],
            addresses_df[addresses_df["address_type"] == "business"]["id"])
    )
    rows = []
    pid = 1
    for (_, acc_row), pname in zip(producer_accounts.iterrows(), producer_names):
        rows.append({
            "id": pid,
            "account_id": int(acc_row["id"]),
            "company_name": pname,
            "company_number": producers_raw[pname]["company_number"],
            "company_description": _producer_descriptions.get(pname, f"{pname} supplier."),
            "lead_time_hours": rng.choice([24, 48, 72]),
            "business_address_id": int(business_addr_by_account[acc_row["id"]]),
        })
        pid += 1
    return pd.DataFrame(rows)


# products table (django catalog.product; fks to producer and category)
# django fields: id, producer_id, category_id, name, description, price, unit,
#                stock, organic_certified, status
def create_products(producers_df: pd.DataFrame) -> pd.DataFrame:
    producer_name_to_id = dict(
        zip(producers_df["company_name"], producers_df["id"])
    )
    rows = []
    prod_id = 1
    for pname, pinfo in producers_raw.items():
        for product_name, pdetails in pinfo["products"].items():
            rows.append({
                "id": prod_id,
                "producer_id": producer_name_to_id[pname],
                "category_id": category_name_to_id[pdetails["category"]],
                "name": product_name,
                "description": f"fresh {product_name} from {pname}.",
                "price": pdetails["price"],
                "unit": pdetails["unit"],
                "stock": random.randint(50, 500),
                "organic_certified": int(random.random() < 0.4),
                "status": "available",
            })
            prod_id += 1
    return pd.DataFrame(rows)


# pick products for one order - mostly staples, occasionally new items
def select_products_for_order(
    staple_products: set[int],
    candidate_product_ids: list[int],
    num_products: int,
) -> set[int]:
    if not candidate_product_ids:
        return set()

    max_unique = min(num_products, len(candidate_product_ids))
    candidate_set = set(candidate_product_ids)
    staple_in_scope = list(staple_products & candidate_set)
    chosen = set()
    while len(chosen) < max_unique:
        if staple_in_scope and random.random() < staple_selection_prob:
            chosen.add(random.choice(staple_in_scope))
        else:
            chosen.add(random.choice(candidate_product_ids))
    return chosen


# reordered flag (ml target). category base rate + loyalty bonus per prior purchase.
def calculate_reordered_flag(
    product_id: int,
    previous_count: int,
    product_category_map: dict[int, str],
) -> int:
    if previous_count == 0:
        return 0
    category = product_category_map[product_id]
    base_prob = category_reorder_prob[category]
    loyalty_bonus = min(loyalty_bonus_step * previous_count, max_loyalty_bonus)
    reorder_prob = min(base_prob + loyalty_bonus, max_reorder_prob)
    return int(np.random.binomial(1, reorder_prob))


# main generation loop - builds orders, producer_orders, order_items, and the ml reorder labels
def generate_all_order_data(
    accounts_df: pd.DataFrame,
    products_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    num_orders: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = []
    producer_orders = []
    order_items = []
    ml_order_products = []

    all_product_ids = products_df["id"].tolist()
    product_name_map = dict(zip(products_df["id"], products_df["name"]))
    product_price_map = dict(zip(products_df["id"], products_df["price"]))
    product_producer_map = dict(zip(products_df["id"], products_df["producer_id"]))
    id_to_category_name = {c["id"]: c["name"] for c in categories_data}
    product_category_map = {
        pid: id_to_category_name[cid]
        for pid, cid in zip(products_df["id"], products_df["category_id"])
    }
    account_to_delivery_addr = dict(
        zip(customers_df["account_id"], customers_df["default_delivery_address_id"])
    )

    # status mix weights
    status_pairs = [(o, p) for (o, p, _) in order_status_mix]
    status_weights = np.array([w for (_, _, w) in order_status_mix])
    status_weights = status_weights / status_weights.sum()

    # customer account ids only
    customer_accounts = accounts_df[accounts_df["account_type"] == "customer"]
    customer_ids = customer_accounts["id"].tolist()

    # timeline setup for realistic growth and customer onboarding.
    month_starts = pd.date_range(base_date, end_date, freq="MS")
    num_months = len(month_starts)
    rng = np.random.default_rng(random_seed + 21)

    # each customer joins at a different month (spread over the 6-year window so
    # the active-user count grows smoothly from 0 rather than starting at full size).
    join_progress = rng.beta(2.2, 2.0, size=len(customer_ids))
    join_month_idx = np.floor(join_progress * (num_months - 1)).astype(int)
    customer_join_month = {
        int(cid): int(mi) for cid, mi in zip(customer_ids, join_month_idx)
    }

    # cumulative active customers per month drives demand: no users -> no orders.
    # this replaces the old flat growth curve and removes the spurious jan-2020 spike.
    active_counts = np.zeros(num_months, dtype=float)
    for mi in join_month_idx:
        active_counts[int(mi):] += 1

    season_factors = np.array([
        1.08 if month_to_season(ts.month) == "summer"
        else 1.05 if ts.month == 12
        else 0.97 if ts.month in (1, 2)
        else 1.00
        for ts in month_starts
    ])
    trend_noise = rng.normal(0, 0.04, num_months)
    monthly_weights = np.maximum(active_counts * season_factors + trend_noise, 0.0)
    # guard against all-zero early months (no active customers yet)
    if monthly_weights.sum() == 0:
        monthly_weights = np.ones(num_months)
    monthly_weights = monthly_weights / monthly_weights.sum()

    # stable preference so some customers order more often than others.
    customer_order_propensity = {
        int(cid): float(rng.lognormal(mean=0.0, sigma=0.45)) for cid in customer_ids
    }

    # allocate each order to a month first, then process orders chronologically.
    order_month_idx = rng.choice(num_months, size=num_orders, p=monthly_weights)
    order_tiebreak = rng.random(num_orders)
    order_sequence = np.lexsort((order_tiebreak, order_month_idx))

    # per-user tracking
    user_order_counter: dict[int, int] = {}
    user_product_history: dict[int, dict[int, int]] = {}
    user_staples: dict[int, set[int]] = {}

    order_id = 1
    producer_order_id = 1
    order_item_id = 1

    for order_slot in order_sequence:
        month_idx = int(order_month_idx[order_slot])
        season = month_to_season(int(month_starts[month_idx].month))

        # only customers who have joined by this month can place this order.
        active_customers = [
            cid for cid in customer_ids if customer_join_month[int(cid)] <= month_idx
        ]
        if not active_customers:
            active_customers = [min(customer_ids, key=lambda c: customer_join_month[int(c)])]

        # customers that have been active longer are slightly more likely to order.
        active_weights = []
        for cid in active_customers:
            tenure_months = max(month_idx - customer_join_month[int(cid)] + 1, 1)
            w = customer_order_propensity[int(cid)] * (1.0 + 0.012 * tenure_months)
            active_weights.append(w)
        active_weights = np.array(active_weights, dtype=float)
        active_weights = active_weights / active_weights.sum()
        acct_id = int(rng.choice(active_customers, p=active_weights))

        # pick order + producer_order status for this order
        status_idx = int(np.random.choice(len(status_pairs), p=status_weights))
        order_status, po_status = status_pairs[status_idx]

        # initialise user state on first order
        if acct_id not in user_staples:
            user_staples[acct_id] = set(
                np.random.choice(all_product_ids, size=num_staple_products, replace=False)
            )
            user_product_history[acct_id] = {}
            user_order_counter[acct_id] = 0

        user_order_counter[acct_id] += 1
        order_number = user_order_counter[acct_id]

        # filter products by seasonal availability.
        available_product_ids = [
            pid for pid in all_product_ids
            if season in product_seasonality.get(product_name_map[pid], ALL_SEASONS)
        ]
        if not available_product_ids:
            available_product_ids = all_product_ids

        # pick products for this order
        num_prods = random.randint(min_products_per_order, max_products_per_order)
        chosen_product_ids = select_products_for_order(
            staple_products=user_staples[acct_id],
            candidate_product_ids=available_product_ids,
            num_products=num_prods,
        )

        # group chosen products by their producer
        producer_product_groups: dict[int, list[int]] = {}
        for pid in chosen_product_ids:
            prod_id = product_producer_map[pid]
            producer_product_groups.setdefault(prod_id, []).append(pid)

        # compute order-level totals
        order_total = Decimal("0.00")
        order_po_rows = []
        order_item_rows = []

        for prod_producer_id, prod_list in producer_product_groups.items():
            po_total = Decimal("0.00")
            po_item_rows = []

            for product_id in prod_list:
                quantity = random.choices([1, 2, 3, 4], weights=[0.50, 0.30, 0.15, 0.05])[0]
                price_snapshot = Decimal(str(product_price_map[product_id]))
                line_total = price_snapshot * quantity
                po_total += line_total

                po_item_rows.append({
                    "id": order_item_id,
                    "producer_order_id": producer_order_id,
                    "product_id": product_id,
                    "quantity": quantity,
                    "price_snapshot": float(price_snapshot),
                    "line_total": float(line_total),
                })
                order_item_id += 1

                # ml: track reordered flag (only for non-cancelled orders)
                prev = user_product_history[acct_id].get(product_id, 0)
                if order_status == "cancelled":
                    reordered = 0
                else:
                    reordered = calculate_reordered_flag(
                        product_id=product_id,
                        previous_count=prev,
                        product_category_map=product_category_map,
                    )
                ml_order_products.append({
                    "order_id": order_id,
                    "product_id": product_id,
                    "reordered": reordered,
                })
                # only count completed/in-flight orders towards loyalty history
                if order_status != "cancelled":
                    user_product_history[acct_id][product_id] = prev + 1

            order_total += po_total

            order_po_rows.append({
                "id": producer_order_id,
                "order_id": order_id,
                "producer_id": prod_producer_id,
                "status": po_status,
                "total_amount": float(po_total),
            })
            producer_order_id += 1
            order_item_rows.extend(po_item_rows)

        commission = float((order_total * Decimal(str(commission_rate))).quantize(Decimal("0.01")))

        orders.append({
            "id": order_id,
            "account_id": acct_id,
            "delivery_address_id": int(account_to_delivery_addr[acct_id]),
            "status": order_status,
            "total_amount": float(order_total.quantize(Decimal("0.01"))),
            "commission_amount": commission,
            "order_number": order_number,
            "planned_month_idx": month_idx,
        })

        producer_orders.extend(order_po_rows)
        order_items.extend(order_item_rows)
        order_id += 1

    return (
        pd.DataFrame(orders),
        pd.DataFrame(producer_orders),
        pd.DataFrame(order_items),
        pd.DataFrame(ml_order_products),
    )


# give each order a created_at timestamp across 2020-2025 (gentle growth trend).
# also derives days_since_prior_order, order_dow, order_hour_of_day for ml.
def assign_order_dates(orders_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    orders_df = orders_df.sort_values(["account_id", "order_number"]).reset_index(drop=True)

    # month start dates for the 5-year window
    month_starts = pd.date_range(base_date, end_date, freq="MS")
    num_months = len(month_starts)

    # use planned month assignments from generation when available.
    if "planned_month_idx" in orders_df.columns:
        month_indices = orders_df["planned_month_idx"].to_numpy(dtype=int)
    else:
        base = np.linspace(1.0, 1.6, num_months)
        noise = rng.normal(0, 0.05, num_months)
        monthly_weights = np.maximum(base + noise, 0.5)
        monthly_weights = monthly_weights / monthly_weights.sum()
        n_orders = len(orders_df)
        month_indices = rng.choice(num_months, size=n_orders, p=monthly_weights)

    # generate a random datetime within each assigned month
    all_dates = []
    for mi in month_indices:
        m_start = month_starts[mi]
        if mi + 1 < num_months:
            m_end = month_starts[mi + 1]
        else:
            m_end = end_date + timedelta(days=1)
        days_in_month = (m_end - m_start).days
        day_offset = rng.integers(0, max(days_in_month, 1))
        hour = rng.integers(order_hour_min, order_hour_max + 1)
        minute = rng.integers(0, 60)
        dt = m_start.to_pydatetime() + timedelta(
            days=int(day_offset), hours=int(hour), minutes=int(minute)
        )
        all_dates.append(dt)

    orders_df["raw_date"] = all_dates
    orders_df = orders_df.sort_values(["account_id", "raw_date"]).reset_index(drop=True)

    # ensure each user's orders are chronologically sorted
    new_dates = []
    for _, group in orders_df.groupby("account_id"):
        sorted_dates = np.sort(group["raw_date"].values)
        new_dates.extend(sorted_dates)

    orders_df["created_at"] = new_dates
    orders_df.drop(columns=["raw_date"], inplace=True)

    # recompute order number from final chronology to keep it consistent with generated dates.
    orders_df = orders_df.sort_values(["account_id", "created_at"]).reset_index(drop=True)
    orders_df["order_number"] = orders_df.groupby("account_id").cumcount() + 1
    orders_df = orders_df.sort_values("id").reset_index(drop=True)

    if "planned_month_idx" in orders_df.columns:
        orders_df = orders_df.drop(columns=["planned_month_idx"])

    # derive ml feature columns
    orders_df = orders_df.sort_values(["account_id", "order_number"]).reset_index(drop=True)
    days_since_prior = []
    order_dows = []
    order_hours = []
    prev_date: dict[int, datetime] = {}

    for _, row in orders_df.iterrows():
        uid = row["account_id"]
        dt = pd.Timestamp(row["created_at"]).to_pydatetime()
        if row["order_number"] == 1:
            days_since_prior.append(np.nan)
        else:
            gap = (dt - prev_date[uid]).days
            days_since_prior.append(max(gap, 1))
        prev_date[uid] = dt
        order_dows.append(dt.weekday())
        order_hours.append(dt.hour)

    orders_df["days_since_prior_order"] = days_since_prior
    orders_df["order_dow"] = order_dows
    orders_df["order_hour_of_day"] = order_hours

    return orders_df.sort_values("id").reset_index(drop=True)


# producer_order dates - match parent order, delivery 2-5 days later (skip if cancelled/pending)
def assign_producer_order_dates(
    producer_orders_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed + 1)
    order_dates = dict(zip(orders_df["id"], orders_df["created_at"]))

    created_dates = []
    delivery_dates = []
    for _, row in producer_orders_df.iterrows():
        order_dt = pd.Timestamp(order_dates[row["order_id"]]).to_pydatetime()
        created_dates.append(order_dt)
        # no delivery date for cancelled / pending / accepted / preparing
        if row["status"] in ("completed", "delivered", "ready"):
            delivery_offset = int(rng.integers(2, 6))
            delivery_dates.append((order_dt + timedelta(days=delivery_offset)).date())
        else:
            delivery_dates.append(None)

    producer_orders_df["created_at"] = created_dates
    producer_orders_df["delivery_date"] = delivery_dates
    return producer_orders_df


# ml feature table - how many times each customer has bought each product,
# plus the date of their most recent purchase (for recency features in the notebook)
def compute_user_product_purchase_counts(
    orders_df: pd.DataFrame,
    ml_order_products_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = ml_order_products_df.merge(
        orders_df[["id", "account_id", "created_at"]],
        left_on="order_id", right_on="id",
    )
    agg = (
        merged.groupby(["account_id", "product_id"])
        .agg(
            user_product_purchase_count=("order_id", "size"),
            last_purchase_date=("created_at", "max"),
        )
        .reset_index()
    )
    return agg


# product_discounts table - ~20% of products carry an active discount (5-25%).
# used in the notebook for the discount-targeting demo.
def compute_product_discounts(
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    active_fraction: float = 0.2,
    min_pct: float = 5.0,
    max_pct: float = 25.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed + 2)
    all_ids = products_df["id"].to_numpy()
    n_active = int(len(all_ids) * active_fraction)
    active_ids = set(rng.choice(all_ids, size=n_active, replace=False))

    latest_order_dt = pd.Timestamp(orders_df["created_at"].max()).to_pydatetime()
    rows = []
    for pid in all_ids:
        active = pid in active_ids
        if active:
            pct = float(round(rng.uniform(min_pct, max_pct), 1))
            start = latest_order_dt - timedelta(days=int(rng.integers(7, 60)))
            end = latest_order_dt + timedelta(days=int(rng.integers(7, 30)))
        else:
            pct = 0.0
            start = None
            end = None
        rows.append({
            "product_id": int(pid),
            "discount_pct": pct,
            "active": bool(active),
            "start_date": start.date() if start else None,
            "end_date": end.date() if end else None,
        })
    return pd.DataFrame(rows)


# write every table to csv under dataset/
def save_dataframes_to_csv(
    dataframes: dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, df in dataframes.items():
        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)


# ====================================================================================================
# run all steps in order
def main() -> None:
    print("[1/12] seeding rng...")
    set_random_seeds(random_seed)

    producer_names = list(producers_raw.keys())

    print("[2/12] creating accounts...")
    accounts_df = create_accounts(num_customers, producer_names)
    print(f"       -> {len(accounts_df)} accounts")

    print("[3/12] creating addresses...")
    addresses_df = create_addresses(accounts_df)
    print(f"       -> {len(addresses_df)} addresses")

    print("[4/12] creating customers...")
    customers_df = create_customers(accounts_df, addresses_df)
    print(f"       -> {len(customers_df)} customers")

    print("[5/12] creating categories...")
    categories_df = create_categories()
    print(f"       -> {len(categories_df)} categories")

    print("[6/12] creating producers...")
    producers_df = create_producers(accounts_df, producer_names, addresses_df)
    print(f"       -> {len(producers_df)} producers")

    print("[7/12] creating products...")
    products_df = create_products(producers_df)
    print(f"       -> {len(products_df)} products")

    print("[8/12] generating orders / producer_orders / order_items / ml labels...")
    orders_df, producer_orders_df, order_items_df, ml_order_products_df = (
        generate_all_order_data(accounts_df, products_df, customers_df, num_orders)
    )
    print(f"       -> {len(orders_df)} orders, {len(producer_orders_df)} producer_orders, "
          f"{len(order_items_df)} order_items, {len(ml_order_products_df)} ml rows")

    print("[9/12] assigning order dates (2020-2025)...")
    orders_df = assign_order_dates(orders_df)
    print(f"       -> date range {orders_df['created_at'].min()} -> {orders_df['created_at'].max()}")

    print("[10/12] assigning producer_order dates + delivery...")
    producer_orders_df = assign_producer_order_dates(producer_orders_df, orders_df)

    print("[11/12] computing user-product purchase counts + product discounts...")
    user_product_counts_df = compute_user_product_purchase_counts(
        orders_df, ml_order_products_df
    )
    print(f"       -> {len(user_product_counts_df)} (user, product) pairs")
    product_discounts_df = compute_product_discounts(products_df, orders_df)
    print(f"       -> {int(product_discounts_df['active'].sum())} active discounts")

    print("[12/12] saving csvs...")
    output_dir = os.path.join(os.path.dirname(__file__), "Dataset")
    save_dataframes_to_csv(
        {
            "accounts": accounts_df,
            "addresses": addresses_df,
            "customers": customers_df,
            "categories": categories_df,
            "producers": producers_df,
            "products": products_df,
            "orders": orders_df,
            "producer_orders": producer_orders_df,
            "order_items": order_items_df,
            "order_products": ml_order_products_df,
            "user_product_purchase_counts": user_product_counts_df,
            "product_discounts": product_discounts_df,
        },
        output_dir,
    )

    print(
        f"dataset generated: {len(accounts_df)} accounts, "
        f"{len(producers_df)} producers, {len(products_df)} products, "
        f"{len(orders_df)} orders, {len(producer_orders_df)} producer_orders, "
        f"{len(order_items_df)} order_items, "
        f"{len(ml_order_products_df)} ml order-product rows, "
        f"{len(product_discounts_df)} product discount rows "
        f"({int(product_discounts_df['active'].sum())} active)"
    )
    print(f"saved to: {output_dir}")
    print("done.")

    # ================================================================================
    # summary of generated tables
    # ================================================================================
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    # columns per table
    tables = {
        "accounts": accounts_df,
        "addresses": addresses_df,
        "customers": customers_df,
        "categories": categories_df,
        "producers": producers_df,
        "products": products_df,
        "orders": orders_df,
        "producer_orders": producer_orders_df,
        "order_items": order_items_df,
        "order_products": ml_order_products_df,
        "user_product_purchase_counts": user_product_counts_df,
        "product_discounts": product_discounts_df,
    }
    print("\n-- tables & columns --")
    for name, df in tables.items():
        print(f"  {name:35s} ({len(df):>6} rows): {list(df.columns)}")

    # categories
    print(f"\n-- categories ({len(categories_df)}) --")
    for _, row in categories_df.iterrows():
        print(f"  {row['id']:>2}. {row['name']}")

    # producers + product counts
    prod_counts = products_df.groupby("producer_id").size()
    print(f"\n-- producers ({len(producers_df)}) --")
    for _, row in producers_df.iterrows():
        n = int(prod_counts.get(row["id"], 0))
        print(f"  {row['id']:>2}. {row['company_name']:25s} | {n} products")

    # products: product | category | producer
    cat_name_by_id = dict(zip(categories_df["id"], categories_df["name"]))
    producer_name_by_id = dict(zip(producers_df["id"], producers_df["company_name"]))
    print(f"\n-- products ({len(products_df)}) --")
    print(f"  {'product':25s} | {'category':12s} | producer")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*20}")
    for _, row in products_df.iterrows():
        print(f"  {row['name']:25s} | {cat_name_by_id[row['category_id']]:12s} | "
              f"{producer_name_by_id[row['producer_id']]}")

    # orders: status + reorder breakdown
    print(f"\n-- orders ({len(orders_df)} orders, {len(ml_order_products_df)} order-product rows) --")
    order_status_counts = orders_df["status"].value_counts()
    print("  order statuses:")
    for s, n in order_status_counts.items():
        print(f"    {s:12s} {n:>6} ({n/len(orders_df):.1%})")
    po_status_counts = producer_orders_df["status"].value_counts()
    print("  producer_order statuses:")
    for s, n in po_status_counts.items():
        print(f"    {s:12s} {n:>6} ({n/len(producer_orders_df):.1%})")
    n_reorder = int(ml_order_products_df["reordered"].sum())
    n_not = int(len(ml_order_products_df) - n_reorder)
    reorder_rate = n_reorder / len(ml_order_products_df)
    print(f"  reordered:     {n_reorder:>6} ({reorder_rate:.1%})")
    print(f"  not reordered: {n_not:>6} ({1-reorder_rate:.1%})")
    print("=" * 80)


if __name__ == "__main__":
    main()