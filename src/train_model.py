import pandas as pd
import yaml
import mlflow
from sklearn.linear_model import LogisticRegression

params = yaml.safe_load(open("params.yaml"))

train = pd.read_csv(params["data"]["selected_path"])

X = train.drop("Attrition", axis=1)
y = train["Attrition"]

with mlflow.start_run():
    model = LogisticRegression(
        max_iter=params["model"]["max_iter"],
        C=params["model"]["C"]
    )
    model.fit(X, y)

    mlflow.log_param("model", params["model"]["name"])
    mlflow.log_param("C", params["model"]["C"])

    mlflow.sklearn.log_model(model, "model")