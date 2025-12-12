import os
from hidden import ngrams_of, filter_real_words as filter_hidden_words
from anagram import do_anagram
from selector import generate_all_selectors, filter_real_words as filter_selector_words
import pandas as pd

import numpy as np
from glove import get_model

model = get_model()

# created a function to get the cosine similarity of 2 word vectorizations


def cosine_sim(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

# this would get the model (GloVe) vectorization of the word


def get_vec(word):
    word = word.lower().strip()
    if word in model.key_to_index:
        return model[word]
    return None

# the would compute the average vector for a defintion that had mulitple
# words since many definitions had multiple words


def avg_vec(text):
    tokens = text.lower().strip().split()
    vecs = [get_vec(t) for t in tokens if get_vec(t) is not None]

    if not vecs:
        return None

    return np.mean(vecs, axis=0)

# this would then return the candidate whose vector has the highest
# average cosine similarity of the defintion. the inputs are the definition
# as a string then a list of candidate words (also strings)


def best_definition_match(definition, candidates):
    dvec = avg_vec(definition)
    if dvec is None:
        return None, {}

    scores = {}
    for w in candidates:
        wvec = get_vec(w)
        scores[w] = cosine_sim(dvec, wvec)

    best_word = max(scores, key=scores.get)
    return best_word, scores


# now the algorithm will run the respective solver when given the categroy

def run_anagram_algorithm(fodder, length):
    words = do_anagram(fodder)
    return {w for w in words if len(w) == length}


def run_hidden_algorithm(fodder, length):
    return set(filter_hidden_words(ngrams_of(length, fodder)))


def run_selector_algorithm(fodder, length):
    return set(filter_selector_words(generate_all_selectors(fodder, length)))

# created an interactive component


def solve_clue():
    print("--- Minute Cryptic Decrypter ---\n")

    fodder = input("Enter the fodder: ").strip()

    while True:
        try:
            length = int(input("Enter the answer length: "))
            break
        except:
            print("Length must be an integer.")

    category = input("Enter category (anagram / hidden / selector): ").lower()

    # generate candidates
    if "anagram" in category:
        candidates = run_anagram_algorithm(fodder, length)
    elif "hidden" in category:
        candidates = run_hidden_algorithm(fodder, length)
    elif "selector" in category:
        candidates = run_selector_algorithm(fodder, length)
    else:
        print("Unknown category.")
        return

    # if it found nothing
    if not candidates:
        print("No English words found.")
        return

    # if it finds one answer it is done
    if len(candidates) == 1:
        print("Solution:", list(candidates)[0])
        return

    # if there are multiple candidates, we have to filter by/ask for the definition
    print(f"\nMultiple candidate words: {candidates}")
    definition = input("Enter the DEFINITION part of the clue: ").strip()

    best, scores = best_definition_match(definition, candidates)

    print("\n--- GloVe Scoring ---")
    print(f"Definition: {definition}")
    print(f"Best Match: {best}\n")

    for w, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{w:<15} {s:.4f}")

    print("\nFinal Answer:", best)


# creating a function to get a csv of this algorithm run on all of the labeled/mislabeled data
def run_batch_on_csv():
    print("\n--- Running solver on testsolver.csv ---")

    ts_path = "testsolver.csv"
    out_ts_path = "testsolver_results.csv"

    try:
        ts_df = pd.read_csv(ts_path)
    except FileNotFoundError:
        print(f"Could not find {ts_path}. Skipping batch solve.\n")
        return

    required_cols = {"Clue", "Category", "Fodder", "Length", "Definition"}
    missing = required_cols - set(ts_df.columns)
    if missing:
        print(f"Missing columns in {ts_path}: {missing}")
        return

    rows = []

    for idx, row in ts_df.iterrows():
        clue = row["Clue"]
        category = row["Category"]
        fodder = row["Fodder"]
        length = int(row["Length"])
        definition = row["Definition"]

        # choose algorithm
        cat = category.lower()
        if "anagram" in cat:
            candidates = run_anagram_algorithm(fodder, length)
        elif "hidden" in cat:
            candidates = run_hidden_algorithm(fodder, length)
        elif "selector" in cat:
            candidates = run_selector_algorithm(fodder, length)
        else:
            candidates = set()

        if not candidates:
            rows.append({
                "clue": clue,
                "category": category,
                "fodder": fodder,
                "length": length,
                "definition": definition,
                "candidate": "",
                "similarity_to_definition": 0.0
            })
            continue

        _, scores = best_definition_match(definition, candidates)

        for cand in candidates:
            rows.append({
                "clue": clue,
                "category": category,
                "fodder": fodder,
                "length": length,
                "definition": definition,
                "candidate": cand,
                "similarity_to_definition": scores.get(cand, 0.0)
            })

    out_df = pd.DataFrame(rows)
    out_ts_path = "testsolver_results.csv"
    full_path = os.path.abspath(out_ts_path)
    out_df.to_csv(out_ts_path, index=False)
    print(f"Saved batch results to {full_path}")


if __name__ == "__main__":
    solve_clue()
  #  run_batch_on_csv()
