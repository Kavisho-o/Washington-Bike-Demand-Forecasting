from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Random train-test split.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def train_model(model, X_train, y_train):
    """
    Fit model.
    """
    model.fit(X_train, y_train)
    return model