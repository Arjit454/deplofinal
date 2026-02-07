import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score

test = pd.read_csv("data/processed/test.csv")

X_test = test.drop("Attrition", axis=1)
y_test = test["Attrition"]

model = mlflow.sklearn.load_model("models:/model/Production")

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("f1_score", f1)