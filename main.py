import os
from src.data_loader import load_data
from src.features import add_cyclical_features, prepare_features
from src.pipeline import build_pipeline
from src.train import split_data, train_model
from src.evaluate import evaluate_model


def main():

    DATA_PATH = "datasets/final/final.csv"

    # Load data
    df = load_data(DATA_PATH)

    # Feature engineering
    df = add_cyclical_features(df)

    # Define feature columns
    cat_cols = ["season", "weathersit", "holiday", "workingday"]
    num_cols = [
        "temp",
        "hum",
        "windspeed",
        "month_sin",
        "month_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "year",
    ]

    # Split features/target
    X, y = prepare_features(df)

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Build pipeline
    model = build_pipeline(cat_cols, num_cols)

    # Train model
    model = train_model(model, X_train, y_train)

    # Evaluate
    results = evaluate_model(model, X_train, y_train, X_test, y_test)

    print("Model Performance:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
