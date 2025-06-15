import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Data generation
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 4 + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Set MLflow experiment
mlflow.set_experiment("Regression_Model_Comparison")


input_example = pd.DataFrame(X_test[:1], columns=["x"])


# --- Linear Regression ---
with mlflow.start_run(run_name="Linear Regression"):
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    preds_lr = model_lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, preds_lr)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("fit_intercept", True)
    mlflow.log_metric("mse", mse_lr)

    signature_lr = infer_signature(X_test, preds_lr)
    mlflow.sklearn.log_model(
        model_lr,
        name="linear_regression_model",
        input_example=input_example,
        signature=signature_lr
    )
    print(f"Linear Regression MSE: {mse_lr:.2f}")


# --- Decision Tree Regression ---
with mlflow.start_run(run_name="Decision Tree Regressor"):
    model_dt = DecisionTreeRegressor(max_depth=5)
    model_dt.fit(X_train, y_train)
    preds_dt = model_dt.predict(X_test)
    mse_dt = mean_squared_error(y_test, preds_dt)

    mlflow.log_param("model_type", "DecisionTreeRegressor")
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("mse", mse_dt)

    signature_dt = infer_signature(X_test, preds_dt)
    mlflow.sklearn.log_model(
        model_dt,
        name="decision_tree_model",
        input_example=input_example,
        signature=signature_dt
    )
    print(f"Decision Tree MSE: {mse_dt:.2f}")