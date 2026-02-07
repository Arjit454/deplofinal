import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))

df = pd.read_csv(params["data"]["raw_path"])

# Encode target
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# Train-test split
train, test = train_test_split(
    df,
    test_size=params["preprocessing"]["test_size"],
    random_state=params["preprocessing"]["random_state"]
)

train.to_csv(params["data"]["train_path"], index=False)
test.to_csv(params["data"]["test_path"], index=False)