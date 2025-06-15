import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 4 + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment 
mlflow.set_experiment("Linear_Regression")


with mlflow.start_run():
    mlflow.log_param("fit_intercept", True)

    #Train 
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_metric("mse", mse)

    input_example = pd.DataFrame(X_test[:1], columns=["x"])
    signature = infer_signature(X_test, y_pred)

    mlflow.sklearn.log_model(
        sk_model=model, 
        name="linear_regression_model",
        input_example=input_example,
        signature=signature
        )

    print(f"Model trained and logged. MSE: {mse:.2f}")

    print(f"Model path: {mlflow.get_artifact_uri('linear_regression_model')}")
