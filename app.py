from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import dotenv
import os
from joblib import load
from numpy.typing import ArrayLike


def load_and_predict(X: ArrayLike, filename: str) -> ArrayLike:
    """
    Deserialize and load the regression model and use it to predict on user provided data.

    This function takes a file name 'filename' that has a default value.
    It uses Joblib 'load' to load the model using the provided file name.
    When the model is loaded, call its `predict` method on provied data.

    Args:
        X (array-like): User provided data used for prediction.
        filename (str): Name of the file that is used to store the model.

    Returns:
        np.ndarray: Predicted value.
    """

    model = load(filename)
    y = model.predict(X)

    return y


def _get_options_from_columns(columns: list, feature_name: str) -> list:
    prefix = f"{feature_name}_"
    options = [col[len(prefix):] for col in columns if col.startswith(prefix)]

    if not options:
        return ["Unknown"]

    return sorted(options)


def create_streamlit_app():
    dotenv.load_dotenv()

    model_path = os.getenv("MODEL_PATH")

    st.title("Salary Prediction Model")
    st.write("Enter your details below to predict the estimated salary.")

    try:
        dir_path = "datasets/"
        X_filename = dir_path + "X.joblib"

        X_train = load(X_filename)

        if not isinstance(X_train, pd.DataFrame):
            st.error("Помилка: X.joblib має бути збережений як pandas DataFrame.")
            return
        expected_columns = X_train.columns
    except FileNotFoundError:
        st.error("Не знайдено файл X.joblib. Переконайся, що він у тій самій папці.")
        return

    employment_opts = _get_options_from_columns(expected_columns, "Employment")
    title_opts = _get_options_from_columns(expected_columns, "Title")
    category_opts = _get_options_from_columns(expected_columns, "Category")
    position_opts = _get_options_from_columns(expected_columns, "Position")
    uses_code_opts = _get_options_from_columns(expected_columns, "Uses_Code")
    language_opts = _get_options_from_columns(expected_columns, "Language")
    company_type_opts = _get_options_from_columns(expected_columns, "Company_Type")
    english_opts = _get_options_from_columns(expected_columns, "English")

    col1, col2 = st.columns(2)

    with col1:
        employment = st.selectbox("Employment", employment_opts)
        title = st.selectbox("Title", title_opts)
        category = st.selectbox("Category", category_opts)
        position = st.selectbox("Position", position_opts)
        uses_code = st.selectbox("Uses Code", uses_code_opts)

    with col2:
        language = st.selectbox("Language", language_opts)
        company_type = st.selectbox("Company Type", company_type_opts)
        english = st.selectbox("English", english_opts)
        experience = st.number_input("Experience (years)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
        age = st.number_input("Age", min_value=15.0, max_value=100.0, value=25.0, step=1.0)

    if st.button("Predict Salary"):
        input_data = pd.DataFrame([{
            "Employment": employment,
            "Title": title,
            "Category": category,
            "Position": position,
            "Uses_Code": uses_code,
            "Language": language,
            "Company_Type": company_type,
            "Experience": experience,
            "English": english,
            "Age": age
        }])

        try:
            input_encoded = pd.get_dummies(input_data)

            input_aligned = input_encoded.reindex(columns=expected_columns, fill_value=0)

            prediction = load_and_predict(input_aligned, filename=model_path)
            predicted_salary = prediction[0]

            st.success(f"Predicted Salary: ${predicted_salary:,.2f}")

            visualize_difference(experience, predicted_salary)

        except FileNotFoundError as e:
            st.error(f"Не знайдено файл моделі: {e.filename}. Перевір шляхи у .env.")
        except Exception as e:
            st.error(f"Сталася помилка під час передбачення: {e}")

def visualize_difference(input_feature: float, prediction: float):
    dir_path = "datasets/"
    X_filename = dir_path + "X.joblib"
    y_filename = dir_path + "y.joblib"

    try:
        X = load(X_filename)
        y = load(y_filename)
    except FileNotFoundError:
        st.warning("Could not find X.joblib or y.joblib to generate the visualization plot.")
        return

    if isinstance(X, pd.DataFrame) and 'Experience' in X.columns:
        X_vis = X['Experience'].values
    else:
        # Fallback: припускаємо, що Experience знаходиться на 8-й позиції (індекс 7)
        X_vis = np.asarray(X)[:, 7] if np.asarray(X).ndim > 1 else np.asarray(X)

    closest_idx = _index_of_closest(X_vis, input_feature)
    actual_target = y[closest_idx]

    difference = actual_target - prediction

    fig = plt.figure(figsize=(8, 5))

    plt.scatter(X_vis, y, color='grey', alpha=0.3, label='Historical Data (Experience vs Salary)')

    plt.scatter(input_feature, actual_target, color='blue', s=100, zorder=5, label=f'Actual Target (Closest)\n${actual_target:,.2f}')

    plt.scatter(input_feature, prediction, color='red', s=100, zorder=5, label=f'Predicted Target\n${prediction:,.2f}')

    plt.plot([input_feature, input_feature], [actual_target, prediction], 'k--', zorder=4)

    mid_point = (actual_target + prediction) / 2
    plt.annotate(f'Diff: ${abs(difference):,.2f}',
                 xy=(input_feature, mid_point),
                 xytext=(input_feature + 0.5, mid_point),
                 arrowprops=dict(arrowstyle="->", color='black'),
                 fontsize=10,
                 color='black')

    plt.legend()
    plt.title("Salary Prediction vs Actual Data based on Experience")
    plt.xlabel("Experience (Years)")
    plt.ylabel("Salary")
    plt.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)

# This is a helper function. No need to edit it
def _index_of_closest(X: ArrayLike, k: float) -> int:
    """
    This function takes an array-like object `X` and a float `k`, and returns the index of the
    element in `X` that is closest to `k`. The function first converts `X` into a NumPy array
    (if it isn't one already) to ensure compatibility with NumPy operations. It then calculates
    the absolute difference between each element in `X` and `k`, identifies the minimum value
    among these differences, and returns the index of this minimum difference.

    Args:
        X (ArrayLike): An array-like object containing numerical data. It can be a list, tuple,
      or any object that can be converted to a NumPy array.
        k (float): The target value to which the closest element in `X` is sought.

    Returns:
        int: The index of the element in `X` that is closest to the value `k`.
    Returns:
        int: Index for the closest value to k in X.
    Finds the index of the element in `X` that is closest to the value `k`.

    """
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == '__main__':
    create_streamlit_app()