from wordfreq import top_n_list
import string

english_words = set(top_n_list("en", 100000))


def generate_permutations(word):
    """
    Return all unique permutations of a word (recursive, no itertools).
    """
    word = word.lower()

    if len(word) <= 1:
        return {word}

    perms = set()

    for i, char in enumerate(word):
        remaining = word[:i] + word[i+1:]
        for p in generate_permutations(remaining):
            perms.add(char + p)

    return perms


def anagrams_of(word):
    """
    Return all raw permutations of the cleaned word.
    """
    cleaned = word.replace(" ", "")
    return generate_permutations(cleaned)


def filter_real_words(candidates):
    """
    Input: set of strings
    Output: subset that are valid English words
    """
    real_words = set()

    for cand in candidates:
        if cand.lower() in english_words:
            real_words.add(cand.lower())

    return real_words


def extend_with_added_letters(words):
    """
    Given a set of valid anagrams,
    return all versions where each letter a–z is either:
        - prepended
        - appended
    plus the original word.
    """
    extended = set()

    for w in words:
        extended.add(w)  # include original

        for letter in string.ascii_lowercase:
            extended.add(letter + w)  # prepend
            extended.add(w + letter)  # append

    return extended


def real_words(word):
    """
    Full pipeline with ONE final output set.
    
    Steps:
    1. generate permutations
    2. filter to real English anagrams
    3. form anagram+1 candidates
    4. filter again
    5. return everything as a single combined set
    """
    # Step 1: all permutations
    perms = anagrams_of(word)

    # Step 2: real anagrams
    real_anas = filter_real_words(perms)

    # Step 3: +1 letter variants
    extended_candidates = extend_with_added_letters(real_anas)

    # Step 4: filter extended ones
    plus_one_real = filter_real_words(extended_candidates)

    # Step 5: combine into one result
    return real_anas.union(plus_one_real)

