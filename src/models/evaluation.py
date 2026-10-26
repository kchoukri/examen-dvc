import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Charger les données de test
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").squeeze("columns")


with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

# Calculer les métriques
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Sauvegarder les prédictions
predictions = pd.DataFrame({
    "valeur_reelle": y_test,
    "prediction": y_pred
})

predictions.to_csv("data/predictions.csv", index=False)

# Sauvegarder les métriques
scores = {
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
    "r2": r2
}

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print("Évaluation terminée avec succès !")
print(scores)