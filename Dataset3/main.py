"""
Dataset 3 - Experiment Runner

Orchestrates the full pipeline:
  1. Preprocessing & EDA  (src/preprocessing.py)
  2. Model training       (src/models/*)
  3. Evaluation & plots   (src/evaluation.py)
"""

from src.preprocessing import run_preprocessing
from src.models.logistic_reg import train_and_evaluate, generate_visualisations


if __name__ == "__main__":
    # -- 1. Preprocessing & EDA --
    df, X, y, splits, total_hours = run_preprocessing()

    # -- 2. Logistic Regression (Algorithm 1) --
    lr_df, last_model, feature_names = train_and_evaluate(splits, X, y)
    generate_visualisations(splits, X, y, lr_df, last_model, feature_names)

    # -- 3. Random Forest (Algorithm 2) --
    # TODO: from src.models.random_forest import ...

    # -- 4. Feed-Forward Neural Network (Algorithm 3) --
    # TODO: from src.models.neural_network import ...
