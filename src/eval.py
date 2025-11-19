import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mae, r2, preds
