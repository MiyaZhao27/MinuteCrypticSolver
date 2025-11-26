import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# import data
df = pd.read_csv("logistic_data.csv")

# create features, right now just length
X = df[["length"]].values.astype(float)

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


def predict_category_from_length(L: float):
    """
    Predict the cryptic clue category using ONLY the length feature.
    L = answer length (float or int)
    """
    x = np.array([[float(L)]])  # can add ore features here
    x_scaled = scaler.transform(x)

    pred_id = logreg.predict(x_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_id])[0]
    pred_probs = logreg.predict_proba(x_scaled)[0]

    return pred_label, pred_probs
