import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Charger les données d'entraînement
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze("columns")

# Définir le modèle
model = RandomForestRegressor(random_state=42)

# Paramètres à tester
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

# Rechercher les meilleurs paramètres
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Sauvegarder les meilleurs paramètres
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)

print("Meilleurs paramètres :", grid_search.best_params_)
print("GridSearch terminé avec succès !")