# Product Reorder Prediction

This project provides a complete pipeline for predicting which products a customer is likely to reorder, based on their purchase history and behaviour patterns.

The system supports:
- Synthetic dataset generation for training and experimentation
- Feature engineering from order history and user behaviour
- Logistic regression model training with a StandardScaler preprocessing pipeline
- Trained model artifact export for deployment in production recommender services

---

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Collaborative Filtering Integration](#collaborative-filtering-integration)
- [Outputs](#outputs)
- [Notes](#notes)

---

## Setup

### 1. Clone the repository

    git clone <repo-url>
    cd <repo-name>

### 2. Install dependencies

    pip install -r requirements.txt

No additional configuration is required. The dataset is synthetic and generated locally.

---

## Project Structure

### dataCreation.py

Generates synthetic training data simulating a grocery e-commerce platform.

Configuration:
- 200 users, 150 products, 8 categories
- Each user places 10–25 orders with 3–8 products per order
- Category-specific reorder probabilities (dairy: 0.75, fruit: 0.60, frozen: 0.45)
- Each user has preferred products selected with 60% probability
- Reorder probability increases as users repurchase the same product

Outputs:

    Dataset/
    ├── users.csv
    ├── user_product_purchase_counts.csv
    ├── products.csv
    ├── orders.csv
    └── order_products.csv
    

---

### LogisticModel.ipynb

Main training notebook covering the full model lifecycle

Key steps:
- Loads and joins the Dataset CSVs
- Creates all 7 features from order history
- Builds data preprocessing pipeline (StandardScaler + OneHotEncoder)
- Trains logistic regression calssifer with balanced class weights
- Evalutates model with accuracy, precision, recall, F1 score, ROC-AUC, confusion matrix
- Includes recommendation function for given user ID.
- Exports the trained model and lookup artifacts to `Models/`

---

### reatrain_with_new_features.py

Extends the base model with two extra features:

- `days_since_last_purchase` - recency signal (0-365 days), taken from the order position in user history 
- `organic_preference` - user-specific fraction of organic product purchases (0-1)

Retrains the full pipeline and saves updated artifacts to `Models/`

---

## Usage 

### 1. Generate synthetic dataset.

```
python dataCreation.py
```

### 2. Model Training
- ### 2a. Notebook method 

    - Open and run `LogisticModel.ipynb` 

- ### 2b. CLI alternative method

    ```
    python retrain_with_new_features.py
    ```


---

## Data Pipeline

1. `dataCreation.py` -> generates synthetic users, products, orders, and order items 
2. `LogisticModel.ipynb` -> joins CSV and creates the features
3. Logistic regression pipeline trained on user-product feature matrix
4. `Models/` -> model and surrounding artifcats saved for deployment


---

## Model Training 

### Features 

Feature | Type | Description
--------|------|------------
user_total_orders | int | Total orders that are placed by a user
product_total_purchases | int | Global purchase count for product
product_reorder_rate | float | Fraction of purchases that were reorders (0-1)
user_product_purchase_count | int | Number of times the specified user has purchased the product
days_since_last_purchase | int | Recency signal (0-365 days)
organic_preference | float | User specific organic product purchase fraction (0-1)
category | string | Product category (one-hot encoded)


### Pipeline 

```
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
])
```

```
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
```
Train/Test split: 80/20 balanced to reflect the overall reorder rate. 


---

## Collaborative Filtering Integration

The trained model is deployed as part of a hybrid recommendation system in the DaESD project. It operates alongside an in-process collaborative filtering (CF) component.


The CF component find users who have bought similar items to the current user, then uses their purcahse history to suggest products.

### How it works:

Each user is represented as a sparse dictionary of products they have bought and
how many times:
```
user_vectors[uid] = {col_index[pid]: count, ...}
```

Two users are considered similar if they have bought many of the same products.
Similarity is measured using cosine similarity — computed manually with no external
ML library:
```
sim = dot(target_vec, other_vec) / (norm(target_vec) * norm(other_vec))
```

The 20 most similar users are selected as neighbours:
```
similarities.sort(key=lambda x: x[0], reverse=True)
top_neighbours = similarities[:20]
```

For each candidate product, the CF score is the weighted average of how often those
neighbours bought it — users who are more similar count for more:
```
for sim, vec in top_neighbours:
    weight = sim / total_sim
    for col_i, count in vec.items():
        cf_scores[col_i] += weight * count
```

This score is then blended with the ML model score to produce the final ranking:
```
Final Score = 0.65 × ML Score  +  0.35 × CF Score
```

### Architecture

The reorder_model.joblib artifact is loaded by a FastAPI microservice. The Django
backend calls this service to get the ML score, then computes the CF score in-process
and blends the two.

- ML Score — the FastAPI /score endpoint receives a 7-feature payload per user-product pair and returns the reorder probability from the loaded pipeline.

- CF Score - computed directly un the Django backend using cosine simiolarity between user puchase vectors:

    - Each user is represented as a sparse dict {product_id: purchase_count}
    - Cosine similarity is calculated manually (no ML library): sparse dot product divided by the product of L2 norms
    - Top 20 most similar users are selected as neighbours
    - CF score for each candidate product = similarity weighted average of neighbour purchase rates
    - Normalised to [0, 1] before blending with the ML score


### Key functions (`recommendations.py` in DaESD backend)

- `get_recommended_products(account, limit)` - main entry point, returns ranked products
- `_score_collaborative(account, available_products)` - CF component
- `_build_user_stats(account)` - builds per-user feature vector
- `_build_global_product_stats` - computes global product purchase counts and re order rates
- `_score_rows(rows)` - passes feature rows to FastAPI through a single HTTP POST request
- ` _generate_reasons(feature_row)` - produces the XAI explaination tags

### XAI tags

Readable reasoning returned alongisde each recommendation:

- "Ordered (N)x by you" - user_product_purchase_count >= 5
- "Previously ordered" - Purchased at least once before by the user
- "Popular repeat purchase" - product_reorder_rate > 0.5
- "Frequently Bought" - product_total_purchases >= 20 

### Purchase Attribution

Recommendation interactions are recorded in the RecommendationInteraction model with a 30-day lookback window. A purchase is attributed to a recommendation if the user viewed or added the product to cart within 30 days. This data supports future model retraining using real purchase signals.


---

## Outputs

    Models/
    ├── reorder_model.joblib                (full trained pipeline: preprocessor + classifier)
    ├── product_purchase_counts.joblib      (dict: product_id -> global purchase count)
    └── product_reorder_rates.joblib        (dict: product_id -> reorder rate 0-1)


The `product_purchase_counts` and `product_reorder_rates` lookup files are used at time of inference to populate the `product_total_purchases` and `product_reorder_rate` features for unseen products without having to re-query the entire dataset.

The notebook also generates predcition CSVs and evaluation plots during training

---

## Notes

The 65/35 ML-CF blend ratio and the 30-day attribution window are configurable in the DaESD backend settings.

Feature-based XAI tags are configurable in `_generate_reasons()`