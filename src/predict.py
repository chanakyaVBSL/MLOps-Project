import joblib
import numpy as np

model = joblib.load('model/model.pkl')

def predict(features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)