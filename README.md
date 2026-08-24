# E-commerce Return-Risk Model

This project trains and evaluates return-risk classifiers on the seeded order
dataset. Run the complete reproducible workflow with:

```bash
python3 -m pip install -r requirements.txt
python3 train_and_evaluate_pt1.py
```

The run writes detailed results to `model_report.json`, full threshold sweeps
to CSV, and the final fitted pipeline to `models/return_risk_model.pkl`.

## Data verification

The dataset has exactly **6,000 rows and 13 columns**. There are 1,365 returned
orders, for an overall return rate of **22.75%**. `rating_given` is missing in
783 rows, or **13.05%**.

| Product category | Rows | Returns | Return rate |
|---|---:|---:|---:|
| Apparel | 1,979 | 523 | 26.43% |
| Beauty | 579 | 116 | 20.03% |
| Electronics | 1,316 | 246 | 18.69% |
| Footwear | 1,071 | 278 | 25.96% |
| Home | 1,055 | 202 | 19.15% |

| Payment method | Rows | Returns | Return rate |
|---|---:|---:|---:|
| COD | 2,501 | 769 | 30.75% |
| Prepaid Card | 1,457 | 245 | 16.82% |
| Prepaid UPI | 1,448 | 245 | 16.92% |
| Wallet | 594 | 106 | 17.85% |

The missingness mechanism is **MAR (missing at random conditional on the
observed `payment_method`)**. The measured missing-rating rate is **22.83% for
COD versus 6.06% for all non-COD orders combined**. It is not MCAR because
missingness depends on payment method, and it is not MNAR because the generator
does not use the unobserved `rating_given` value to decide whether to hide it.

## Leakage-safe preprocessing

Raw data is split into a stratified 80/20 train/test split with
`random_state=42` before preprocessing. A `ColumnTransformer` inside each
scikit-learn `Pipeline` median-imputes and standard-scales numeric features,
and mode-imputes and one-hot encodes `product_category` and `payment_method`.
It is fitted on the 4,800 training rows only; the 1,200 test rows are only
passed to `transform`. `order_id` and `returned` are excluded from predictors.

## Dummy baseline

| Accuracy | F1 (`returned=1`) | Recall (`returned=1`) |
|---:|---:|---:|
| 77.25% | 0.0000 | 0.00% |

The apparently high accuracy is misleading because the most-frequent dummy
always predicts “not returned.” This is the **high accuracy, zero recall** trap:
it misses every return, the outcome the business needs to detect. Honest
evaluation therefore compares against a baseline and uses F1/recall metrics
aligned with return detection instead of accuracy alone.

## Logistic Regression

The model uses `class_weight="balanced"`. At the default 0.50 threshold:

| Accuracy | F1 | Recall | Precision | ROC-AUC |
|---:|---:|---:|---:|---:|
| 59.17% | 0.3921 | 57.88% | 29.64% | 0.6253 |

The threshold was swept from 0.10 through 0.90 in increments of 0.01. The full
81-row result is in `logistic_threshold_sweep.csv`; representative F1 values
are tabulated here:

| Threshold | 0.10 | 0.20 | 0.30 | 0.40 | **0.44** | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | .3707 | .3707 | .3762 | .4011 | **.4091** | .3921 | .3225 | .1288 | .0000 | .0000 |

The F1-maximising threshold is **0.44**, with **75.82% recall** and **28.01%
precision**. Relative to 0.50, recall rises by **17.95 percentage points** and
precision falls by **1.63 points**. Lowering the threshold makes missed returns
(false negatives) more expensive to avoid, while accepting more false alarms
(false positives) and their review/intervention cost.

## Tuned Random Forest

`GridSearchCV` used ROC-AUC scoring and five-fold shuffled `StratifiedKFold`
cross-validation. It tested `n_estimators` in `[100, 200]` and `max_depth` in
`[6, 10, None]` with `class_weight="balanced"` and `random_state=42`.

| Best `n_estimators` | Best `max_depth` | Best CV ROC-AUC | Test ROC-AUC | Absolute gap |
|---:|---:|---:|---:|---:|
| 200 | 6 | 0.6192 | 0.6203 | 0.0011 |

The tiny CV/test gap is evidence against severe overfitting. Repeating the
threshold sweep on this Random Forest's own held-out `predict_proba` output
gives **t\*_rf = 0.50**, with F1 0.4076, recall 54.95%, and precision 32.40%.
Its full sweep is in `random_forest_threshold_sweep.csv`; it is deliberately
not calibrated using the Logistic Regression threshold.

## Feature importance and explanation

Permutation importance uses held-out ROC-AUC with 20 deterministic repeats.

| Feature | Impurity importance (rank) | Permutation importance (rank within top 5) |
|---|---:|---:|
| `payment_method_COD` | 0.1788 (1) | 0.0651 (1) |
| `price_inr` | 0.1323 (2) | 0.0124 (2) |
| `delivery_distance_km` | 0.0957 (3) | 0.0006 (4) |
| `customer_tenure_days` | 0.0900 (4) | -0.0051 (5) |
| `delivery_days` | 0.0884 (5) | 0.0030 (3) |

COD plausibly raises risk because the generator explicitly assigns it higher
return log-odds. Higher-priced purchases can prompt more scrutiny and price is
also in the generating equation. Tenure captures customer familiarity and is
explicitly protective in the generator. Slower delivery may increase
dissatisfaction and is explicitly included in return risk. Distance might look
plausible as a delivery-friction proxy, but it is absent from the true equation.

`delivery_distance_km` and especially `customer_tenure_days` lose most of their
apparent value under permutation (distance falls from 0.0957 to 0.0006 and
tenure to a negative test-set importance). Impurity importance can overrate a
noisy continuous feature because its many possible split points give a tree
more chances to find accidental training-set improvements; held-out
permutation importance exposes whether those splits generalise.

## Subgroup analysis

Metrics use the selected Random Forest and t\*_rf = 0.50. Overall recall is
54.95% and precision is 32.40%.

| Product category | Test rows | Actual returns | Recall | Precision |
|---|---:|---:|---:|---:|
| Apparel | 385 | 100 | 52.00% | 31.71% |
| Beauty | 116 | 31 | 61.29% | 47.50% |
| Electronics | 261 | 52 | 44.23% | 32.86% |
| Footwear | 217 | 56 | 58.93% | 36.26% |
| Home | 221 | 34 | 67.65% | 23.47% |

| Payment method | Test rows | Actual returns | Recall | Precision |
|---|---:|---:|---:|---:|
| COD | 503 | 155 | 93.55% | 32.73% |
| Prepaid Card | 283 | 49 | 2.04% | 20.00% |
| Prepaid UPI | 294 | 48 | 4.17% | 33.33% |
| Wallet | 120 | 21 | 9.52% | 22.22% |

Electronics has meaningfully weaker recall than average (44.23% versus 54.95%),
and prepaid methods are much weaker still. A concrete next step is to tune
separate validation-set thresholds for COD and each prepaid method, lowering
the prepaid thresholds to recover recall while constraining precision to an
acceptable intervention-cost floor.

## Saved artifact

`models/return_risk_model.pkl` is the fitted winning pipeline containing both
the preprocessing `ColumnTransformer` and the tuned
`RandomForestClassifier(max_depth=6, n_estimators=200)`. The training script
reloads it with `joblib.load` and verifies that its `predict_proba` output is
identical before reporting success.
