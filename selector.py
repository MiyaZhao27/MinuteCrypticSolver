from word2vec import get_model
import re

model = get_model()

anchors = {
    "FIRST": "first",
    "LAST": "last",
    "MIDDLE": "middle",
    "ODD": "odd",
    "EVEN": "even",
    "INTERMITTENT": "skip",
    "NTH": "nth"  
}
NUMBER_WORDS = {
    "one": 1, "first": 1, "once": 1, "primary": 1, "single": 1,
    "two": 2, "second": 2, "twice": 2, "binary": 2,
    "three": 3, "third": 3, "triple": 3, "tertiary": 3,
    "four": 4, "fourth": 4, "quadrant": 4, "quaternary": 4,
    "five": 5, "fifth": 5, "quinary": 5,
    "six": 6, "sixth": 6,
    "seven": 7, "seventh": 7,
    "eight": 8, "eighth": 8,
    "nine": 9, "ninth": 9,
    "ten": 10, "tenth": 10
}


def resolve_canonical(indicator):
    """
    Use word2vec similarity to guess which canonical selector the clue means.
    """
    scores = {}
    for canon, anchor in anchors.items():
        try:
            scores[canon] = model.similarity(indicator.lower(), anchor)
        except KeyError:
            scores[canon] = -1
            
    best = max(scores, key=scores.get)
    return best

def detect_n_from_indicator(indicator, model):
    """Try to determine N if the selector is NTH."""
    indicator = indicator.lower()

    m = re.search(r"\d+", indicator)
    if m:
        return int(m.group(0))

    # textual numbers
    for word, n in NUMBER_WORDS.items():
        if word in indicator:
            return n

    best_n = None
    best_sim = -1
    for word, n in NUMBER_WORDS.items():
        try:
            sim = model.similarity(indicator, word)
        except KeyError:
            sim = -1
        if sim > best_sim:
            best_sim = sim
            best_n = n

    return best_n 


def apply_indicator(indicator, fodder, length=None, model=None):
    """
    Main selection logic — including W2V-driven selector detection.
    """
    words = fodder.split()
    combined = "".join(words)

    # Determine canonical selector rule using Word2Vec
    if model is None:
        raise ValueError("You must supply a Word2Vec model.")

    canon = resolve_canonical(indicator, model)

    # ================== all basic ones Eliza-Style =============
    if canon == "FIRST":
        result = [w[0] for w in words]
        return "".join(result[:length]) if length else "".join(result)

    if canon == "LAST":
        result = [w[-1] for w in words]
        return "".join(result[:length]) if length else "".join(result)

    if canon == "MIDDLE":
        result = [w[len(w)//2] for w in words if len(w) > 2]
        return "".join(result[:length]) if length else "".join(result)

    if canon == "EVEN":
        result = []
        for w in words:
            for c in w[1::2]:
                result.append(c)
                if length and len(result) == length:
                    return "".join(result)
        return "".join(result)

    if canon == "ODD":
        result = []
        for w in words:
            for c in w[0::2]:
                result.append(c)
                if length and len(result) == length:
                    return "".join(result)
        return "".join(result)

    if canon == "INTERMITTENT":
        result = []
        for w in words:
            for c in w[0::2]:
                result.append(c)
                if length and len(result) == length:
                    return "".join(result)
        return "".join(result)

    if canon == "NTH":
        N = detect_n_from_indicator(indicator, model)
        ## in-case
        if N is None: 
            return combined if length is None else combined[:length]

        result = []
        for w in words:
            for c in w[::N]:
                result.append(c)
                if length and len(result) == length:
                    return "".join(result)
        return "".join(result)
    
    
### in-case --> use word2vec to generate list of words similar to definition and then choose based on letter length?

