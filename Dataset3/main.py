from src.models.logistic_reg import generate_visualisations, train_and_evaluate
from src.preprocessing import run_preprocessing


def main():
    data, features, target, splits, total_hours = run_preprocessing()
    results, last_model, feature_names = train_and_evaluate(splits, features, target)
    generate_visualisations(splits, features, target, results, last_model, feature_names)


if __name__ == "__main__":
    main()