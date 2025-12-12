
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from glove import get_model

# load in our labeled data containing category, clue, indicator, fodder, definition,
# and length of clue (these are all the hints provided to a human player by Minute Cryptic)

data_path = "logistic_data.csv"

df = pd.read_csv(data_path)

# load our word vectorization model (GloVe 50D trained on tweets)

print("Loading GloVe model...")
model = get_model()
print("GloVe model loaded.")

# like with many previous papers, we compared the cosine similarities of vectors to
# determine semantic similarity (dot product of the 2 vectors/their norms multiplied)
# we also had it handle 0 to prevent erroring


def cosine_sim(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

# we defined a function to return the GloVe vector for a word
# it returns None if they can't find the vector


def get_vec(word):
    if word is None:
        return None
    word = str(word).lower()
    if word in model.key_to_index:
        return model[word]
    return None

# we wanted to get cosine similarities to known indicators for the anagrams, hiddens
# and selectors. We defined the word bank from the Minute Cryptic instructions on
# how to play the game


ANAGRAM_WORDS = [
    "mix", "throwing", "destroy", "strange",
    "dancing", "sort", "tampering", "exploded"
]

HIDDEN_WORDS = [
    "hides", "displays", "reveals", "within", "held",
    "capturing", "absorbed", "sample", "selection", "bit", "taken"
]

SELECTOR_WORDS = [
    "head", "tail", "heart", "borders", "coat",
    "contents", "guts", "odd", "even", "alternate", "regularly"
]

# we eventually wanted to get the average similarity between an indicator
# and the list of refrences words from the above banks to add as features
# this functioned allowed us to average the cosine similarities of the indicator
# to each of the words in each of the banks


def avg_similarity(indicator, word_list):
    ivec = get_vec(indicator)
    sims = [cosine_sim(ivec, get_vec(w)) for w in word_list]
    return float(np.mean(sims)) if sims else 0.0

# then we went on to add the features

# feature 1: fodder length. we thought selectors might have longer fodders
# especially if you're going to take the first/last letters of everything
# first we cleaned it slightly


df["fodder_length"] = (
    df["fodder"]
    .astype(str)
    .str.replace(r"[^A-Za-z]", "", regex=True)
    .str.len()
)

# feature 2: number of words in fodder. this had a similar modivaton-- perhaps
# more convoluted hints are characteristic of certain puzzles

df["fodder_word_count"] = df["fodder"].astype(str).str.split().apply(len)

# features 3-5: these were the average cosine similarites to the bank of words

df["glove_anagram"] = df["indicator"].apply(
    lambda x: avg_similarity(str(x), ANAGRAM_WORDS)
)
df["glove_hidden"] = df["indicator"].apply(
    lambda x: avg_similarity(str(x), HIDDEN_WORDS)
)
df["glove_selector"] = df["indicator"].apply(
    lambda x: avg_similarity(str(x), SELECTOR_WORDS)
)

# feature 6: we also added length of answer for similar modivations

# feature matrix
FEATURE_COLS = [
    "length",
    "fodder_length",
    "fodder_word_count",
    "glove_anagram",
    "glove_hidden",
    "glove_selector",
]

X = df[FEATURE_COLS].values.astype(float)
y_raw = df["category"].values

# encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)


# due to our small sample side, we wanted to cross-validate

print("\n ----- Cross-Validated Evaluation (5-fold Stratified) ------")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []

fold_idx = 1
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler_cv = StandardScaler()
    X_train_scaled = scaler_cv.fit_transform(X_train)
    X_test_scaled = scaler_cv.transform(X_test)

    # we wanted to use multinomial regression
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    fold_idx += 1

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)


# to report the findings on the labled data ...
print("\n----- Classification Report (aggregated over folds) -----")
print(classification_report(
    all_y_true,
    all_y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

print("\n-----Confusion Matrix (aggregated over folds) -----")
cm = confusion_matrix(all_y_true, all_y_pred)
print(cm)

print("\nLabels (row = true, col = pred):", list(label_encoder.classes_))


# we then had to finalize the model to be able to access user imputed
# indicators to crack

# we trained the scaler and model on the full dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logreg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500
)
logreg.fit(X_scaled, y)

print("\nTrained on all data.")


# to build an interactive component we had to extract what the user inputed
# this function builds a feature matrix like we did for the training data

def extract_features(clue, indicator, length, fodder, definition):
    # Clean fodder: count only letters
    fodder_clean_len = len("".join([c for c in str(fodder) if c.isalpha()]))

    return np.array([[
        float(length),
        float(fodder_clean_len),
        len(str(fodder).split()),
        avg_similarity(indicator, ANAGRAM_WORDS),
        avg_similarity(indicator, HIDDEN_WORDS),
        avg_similarity(indicator, SELECTOR_WORDS)
    ]])

# this function predicts the probability distribution for each set of
# user inputed clues


def predict_category(clue, indicator, length, fodder, definition):
    x = extract_features(clue, indicator, length, fodder, definition)
    x_scaled = scaler.transform(x)

    pred_id = logreg.predict(x_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_id])[0]
    pred_probs = logreg.predict_proba(x_scaled)[0]

    return pred_label, pred_probs

# This part prompts the user


print("\n----- What Category is the Minute Cryptic Indicator? -----")

try:
    clue = input("Enter the clue: ")
    indicator = input("Enter the indicator: ")
    fodder = input("Enter the fodder words: ")
    definition = input("Enter the definition: ")
    length = int(input("Enter the solution length: "))

    pred_label, pred_probs = predict_category(
        clue, indicator, length, fodder, definition
    )

    print("\n----- Prediction Result -----")
    print("Predicted Category:", pred_label)

    print("\nProbabilities:")
    for cat, p in zip(label_encoder.classes_, pred_probs):
        print(f"  {cat}: {p:.4f}")

except KeyboardInterrupt:
    print("\nExiting interactive mode.")


# Printing Out the results of all of our training data for analytical use
# we have it commented out to prevent duplicate downloads
# results = []

# for idx, row in df.iterrows():
#     clue = row["clue"]
#     indicator = row["indicator"]
#     fodder = row["fodder"]
#     definition = row["definition"]
#     length = row["length"]
#     true_cat = row["category"]

#     # build features like training
#     x = extract_features(clue, indicator, length, fodder, definition)
#     x_scaled = scaler.transform(x)

#     # get probabilities and predicted label
#     prob_vec = logreg.predict_proba(x_scaled)[0]
#     pred_id = np.argmax(prob_vec)
#     pred_cat = label_encoder.inverse_transform([pred_id])[0]

#     # store prob for each category
#     prob_dict = {
#         f"prob_{cat}": prob_vec[i]
#         for i, cat in enumerate(label_encoder.classes_)
#     }

#     # store results
#     results.append({
#         "clue": clue,
#         "true_category": true_cat,
#         "predicted_category": pred_cat,
#         **prob_dict
#     })

# probs_df = pd.DataFrame(results)

# print("\n----- Per-clue Probabilities -----")
# print(probs_df.head())

# # Save to CSV
# output_path = "clue_probabilities.csv"
# probs_df.to_csv(output_path, index=False)
# print(f"\nSaved per-clue probabilities to {output_path}")
