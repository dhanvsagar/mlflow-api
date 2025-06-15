# Linear Regression with MLflow Tracking and Model Serving
A minimal example of training a linear regression model using scikit-learn, logging the experiment and model using MLflow, and serving the trained model as a REST API.


## Requirements

- Install the pip packages:

```bash
pip install mlflow scikit-learn pandas numpy
```

## Steps to Run

-  Train the Model and Log with MLflow
```
python mlflow_linear_regression.py
```

- Start the MLflow Tracking UI
```
mlflow ui --port 6006
```

-  Serve the Trained Model as REST API
```
mlflow models serve -m mlruns/0/<run_id>/artifacts/linear_regression_model -p 1234 --no-conda 
```

(You can also get the url from the mlflow UI under Artifacts)


- Make Predictions via API 

```
curl -X POST http://127.0.0.1:1234/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["x"],
      "data": [[2.5], [7.0]]
    }
  }'
```

- OR use python requests example `request_eg.py`