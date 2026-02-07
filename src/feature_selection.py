import pandas as pd
import yaml

params = yaml.safe_load(open("params.yaml"))

train = pd.read_csv(params["data"]["train_path"])

# Correlation-based feature selection
corr = train.corr()["Attrition"].abs()
selected_features = corr[corr > params["feature_selection"]["correlation_threshold"]].index

train[selected_features].to_csv(params["data"]["selected_path"], index=False)