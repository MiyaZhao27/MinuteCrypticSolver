from hidden import ngrams_of, filter_real_words
from anagram import anagrams_of, filter_real_words
from selector import generate_all_selectors, filter_real_words


def run_anagram_algorithm(fodder, length):
    all_anagrams = anagrams_of(fodder)
    real_words = filter_real_words(all_anagrams)

    def filter_by_length(words, length):
        return {w for w in words if len(w) == length}

    real_words = filter_by_length(real_words, length)

    if len(real_words) == 0:
        return f"No anagram English words of length {length} found."

    return f"Anagram English words of length {length}: {real_words}"


def run_hidden_algorithm(fodder, length):
    all_ngrams = ngrams_of(length, fodder)
    real_words = filter_real_words(all_ngrams)

    if len(real_words) == 0:
        return f"No hidden English words of length {length} found."

    return f"Hidden English words of length {length}: {real_words}"


def run_selector_algorithm(fodder, length):
    candidates = generate_all_selectors(fodder, length)
    real = filter_real_words(candidates)

    if len(real) == 0:
        return f"No selector-based English words of length {length} found."

    return f"Selector English words of length {length}: {real}"


def solve_clue():
    print("=== Minute Cryptic Decrypter ===\n")

    fodder = input("Enter the fodder words: ")

    while True:
        try:
            length = int(input("Enter the solution length: "))
            break
        except:
            print("Length must be an integer. Try again.")

    category = input(
        "Enter the category (anagram / hidden / selector): ").lower()

    print(f"\nUsing category: {category}")

    if "anagram" in category:
        print(run_anagram_algorithm(fodder, length))

    elif "hidden" in category:
        print(run_hidden_algorithm(fodder, length))

    elif "selector" in category:
        print(run_selector_algorithm(fodder, length))

    else:
        print("Unknown category — sorry.")


if __name__ == "__main__":
    solve_clue()
