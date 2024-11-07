import pandas as pd
import joblib
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

train = pd.read_csv("data/train.csv")

X = train.drop(columns=["id", "FloodProbability"])
y = train["FloodProbability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_params = {
    'max_depth': 9,
    'n_estimators': 913,
    'gamma': 0.007424095823836917,
    'reg_alpha': 0.2144593472679007,
    'reg_lambda': 1.7404175052607878,
    'min_child_weight': 0,
    'subsample': 0.21959794108855646,
    'colsample_bytree': 0.5971654266907475,
    'learning_rate': 0.6343746303785935
}

lgb = LGBMRegressor()
cat = CatBoostRegressor(silent=True)

model = StackingRegressor(
    estimators=[("cat", cat), ("lgb", lgb)],
    final_estimator=XGBRegressor(**xgb_params)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

joblib.dump(model, 'model/model.pkl')