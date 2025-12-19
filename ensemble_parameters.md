# Ensemble Method Parameters

This document details the specific parameters and configurations used for each of the ensemble methods implemented in `test_Ground1.py`.

---

## 1. Simple Average

-   **Description:** The probabilities for each class are averaged across all base models. The final prediction is the class with the highest average probability.
-   **Parameters:** None.

---

## 2. Hard Voting

-   **Description:** Each base model makes a prediction (the class with the highest probability). The final prediction is the class that receives the most votes from the base models.
-   **Parameters:** None.

---

## 3. Weighted Average

-   **Description:** A weighted average of the probabilities from the base models is calculated. The weights are optimized to minimize the log-loss with respect to the true labels.
-   **Optimization:**
    -   **Method:** `scipy.optimize.minimize` (using the default L-BFGS-B algorithm).
    -   **Objective Function:** Minimize the `log_loss` between the weighted probabilities and the true labels.
    -   **Constraints:**
        -   The sum of all model weights must equal 1.
        -   Each individual weight must be between 0 and 1.

---

## 4. Rank Averaging

-   **Description:** For each class, the probabilities from each model are converted to ranks. The ranks for each class are then summed up across all models. The final prediction is the class with the highest total rank.
-   **Parameters:** None.

---

## 5. Geometric Mean

-   **Description:** The geometric mean of the probabilities for each class is calculated across all base models.
-   **Parameters:** None.

---

## 6. Stacking with Logistic Regression

-   **Description:** A meta-model is trained on the outputs (probabilities) of the base models.
-   **Meta-Model:** `sklearn.linear_model.LogisticRegression`
-   **Model Parameters:**
    -   `multi_class='multinomial'`: The loss function is the multinomial loss that handles multiple classes.
    -   `max_iter=1000`: The maximum number of iterations for the solver to converge.

---

## 7. Stacking with Random Forest

-   **Description:** A Random Forest classifier is used as the meta-model, trained on the probabilities of the base models.
-   **Meta-Model:** `sklearn.ensemble.RandomForestClassifier`
-   **Model Parameters:**
    -   `n_estimators=100`: The number of trees in the forest.
    -   `max_depth=5`: The maximum depth of each tree.
    -   `random_state=42`: Ensures reproducibility of the results.

---

## 8. Stacking with Gradient Boosting

-   **Description:** A Gradient Boosting classifier is used as the meta-model, trained on the probabilities of the base models.
-   **Meta-Model:** `sklearn.ensemble.GradientBoostingClassifier`
-   **Model Parameters:**
    -   `n_estimators=100`: The number of boosting stages to perform.
    -   `max_depth=3`: The maximum depth of the individual regression estimators.
    -   `random_state=42`: Ensures reproducibility of the results.
