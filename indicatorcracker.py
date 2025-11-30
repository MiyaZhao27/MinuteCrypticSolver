import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# import data
df = pd.read_csv("logistic_data.csv")





# FEATURES HERE!!!!

# Fodder Length
## NEW

df['fodder_length'] = df['fodder'].apply(lambda s: len(s) if isinstance(s, str) else 0)

df["fodder_length"] = (df["fodder"]
    .astype(str)
    .str.replace(r"[^A-Za-z]", "", regex=True) 
    .str.len()
)


# Length of Clue and Fodder
X = df[["length", "fodder_length"]].values.astype(float)

# label categories
y_raw = df["category"].values

# encode label for model
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    stratify=y,
    random_state=42
)
# scale it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# train the model classifier
logreg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500
)

logreg.fit(X_train_scaled, y_train)

# evaluate the performance

y_pred = logreg.predict(X_test_scaled)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# have it spit out the categrory for solver


def predict_category_from_length(L: float, F: float):
    """
    Predict the cryptic clue category using the length and Fodder Length feature.
    L = answer length (float or int)
    F = Fodder length (Float)
    """
    x = np.array([[float(L), float(F)]])
    x_scaled = scaler.transform(x)

    pred_id = logreg.predict(x_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_id])[0]
    pred_probs = logreg.predict_proba(x_scaled)[0]

    return pred_label, pred_probs
