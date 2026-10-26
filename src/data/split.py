import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Charger le dataset
df = pd.read_csv("data/raw_data/raw.csv")

# Séparer les variables et la cible
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Path("data/processed_data").mkdir(parents=True, exist_ok=True)

# Sauvegarder les 4 fichiers
X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)

print("Split terminé avec succès !")