from __future__ import annotations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from numpy.typing import ArrayLike
from joblib import dump, load
import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler


def load_datasets(dir_path, X_path, y_path):
    X = load(dir_path + X_path)
    y = load(dir_path + y_path)

    return X, y

def prepare_datasets(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    float_cols = ['Experience', 'Age']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), float_cols),
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    final_model = TransformedTargetRegressor(
        regressor=model_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    return X_train, X_test, y_train, y_test, final_model


def evaluate_regression_model(model, X_test: ArrayLike, y_test: ArrayLike):
    """
    Evaluate the performance of a regression model on test data.

    This function takes a trained regression 'model', test feature matrix 'X_test',
    and corresponding test target values 'y_test'. It calculates Mean Squared Error (MSE)
    and prints it in terminal.

    Args:
        model (sklearn.linear_model.LinearRegression): Trained regression model to be evaluated.
        X_test (array-like): Test feature matrix.
        y_test (array-like): Validation target values.

    """

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")


if __name__ == '__main__':
    dir_path = "datasets/"
    X_path = "X.joblib"
    y_path = "y.joblib"

    X, y = load_datasets(dir_path, X_path, y_path)

    X_train, X_test, y_train, y_test, model = prepare_datasets(X, y)

    model.fit(X_train, y_train)

    evaluate_regression_model(model, X_test, y_test)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_path = f"models/model_{timestamp}.joblib"

    dump(model, model_path)
