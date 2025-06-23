import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(csv_path, random_state = 42):
    """
    Loads dataset, selects features, splits into train/test sets.
    Returns: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(csv_path)

    features = ["predicted", "avg_temp", "avg_humidity", "avg_co2", "avg_tvoc"]
    target = "label"

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
