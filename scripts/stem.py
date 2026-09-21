"""
The Porter stemmer, and why every word in this study goes through it.

── The bug this exists to fix ───────────────────────────────────────────────
Every word-counting metric here did exact string matching, so `screens` and
`screening` were two different words. On a real server in the corpus:

    analyze_job_description  "Extract what a job posting actually screens on."
    optimize_resume          "Rewrite a resume so it passes ATS screening."

those two tools shared their one genuinely linking term and the distinctive-
share counter scored both of them as distinctive on it. That is wrong in one
direction only — a missed merge always *raises* distinctiveness and *lowers*
the collision rate — so every published figure was optimistic.

── Why a stemmer written out here rather than installed ────────────────────
`requirements.txt` has one line in it, and `03_analyse.py` is meant to run on
a bare Python 3.12. More importantly the checker at getmcpulse.com/check runs
the same method in the reader's browser, in TypeScript, and prints its answer
beside a corpus figure computed by this file. Two stemmers from two ecosystems
that agree on 99% of words would put a silent, unattributable difference
between the two numbers. So there is one algorithm, written twice, held
together by `data/stem_fixture.json` — see `tests/stem.test.ts` in the web repo.

This is Porter (1980) as published, with one deviation, below.

── The deviation: words of three letters or fewer are left alone ────────────
Porter's own implementation stops at two. Three is here because this corpus is
full of three-letter acronyms and Porter treats their trailing `s` as a plural:
`ats` -> `at`, `ids` -> `id`, `api`/`apis` likewise. `ATS` is the single most
diagnostic noun on the job-applications server this whole line of work came
from, and turning it into `at` — which is also a stopword — would delete the
finding rather than sharpen it.

The cost is that `ids` and `id` stay separate. That is a missed merge, which
errs towards under-reporting, the direction everything in this study errs in.
"""

VOWELS = frozenset("aeiou")

#: Below this length a word is returned unchanged. See the module docstring.
MIN_STEM_LENGTH = 4


def _consonant(word, i):
    """Porter's definition: a letter other than a vowel, and other than `y`
    preceded by a consonant."""
    letter = word[i]
    if letter in VOWELS:
        return False
    if letter == "y":
        return i == 0 or not _consonant(word, i - 1)
    return True


def _measure(stem):
    """`m` — the number of vowel-consonant sequences in [C](VC)^m[V]."""
    m = 0
    i = 0
    n = len(stem)

    while i < n and _consonant(stem, i):
        i += 1
    while i < n:
        while i < n and not _consonant(stem, i):
            i += 1
        if i == n:
            break
        m += 1
        while i < n and _consonant(stem, i):
            i += 1
    return m


def _has_vowel(stem):
    return any(not _consonant(stem, i) for i in range(len(stem)))


def _double_consonant(stem):
    return len(stem) >= 2 and stem[-1] == stem[-2] and _consonant(stem, len(stem) - 1)


def _cvc(stem):
    """Ends consonant-vowel-consonant, the last not `w`, `x` or `y`."""
    n = len(stem)
    if n < 3:
        return False
    return (
        _consonant(stem, n - 3)
        and not _consonant(stem, n - 2)
        and _consonant(stem, n - 1)
        and stem[-1] not in "wxy"
    )


def _replace(word, rules, min_measure):
    """First matching rule wins, longest suffix first.

    Sorted by length rather than left in Porter's letter-switched order, which
    is an implementation detail of his C rather than part of the algorithm.
    Longest-first gives the same answers — `ization` before `ation`, `ement`
    before `ment` before `ent` — and is the one property the TypeScript twin
    has to reproduce, so it is stated rather than implied.
    """
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > min_measure:
                return stem + replacement
            return word
    return word


_STEP2 = sorted(
    [
        ("ational", "ate"), ("tional", "tion"),
        ("enci", "ence"), ("anci", "ance"),
        ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
        ("eli", "e"), ("ousli", "ous"),
        ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
        ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ],
    key=lambda rule: -len(rule[0]),
)

_STEP3 = sorted(
    [
        ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
        ("ical", "ic"), ("ful", ""), ("ness", ""),
    ],
    key=lambda rule: -len(rule[0]),
)

_STEP4 = sorted(
    [
        ("al", ""), ("ance", ""), ("ence", ""), ("er", ""), ("ic", ""),
        ("able", ""), ("ible", ""), ("ant", ""), ("ement", ""), ("ment", ""),
        ("ent", ""), ("ou", ""), ("ism", ""), ("ate", ""), ("iti", ""),
        ("ous", ""), ("ive", ""), ("ize", ""),
    ],
    key=lambda rule: -len(rule[0]),
)


#: Every word the corpus has already stemmed.
#:
#: The study makes nine passes over 82,549 tools — two pools, the union, and six
#: size buckets — re-tokenising every description each time, so the same few
#: hundred thousand distinct words get stemmed tens of millions of times. Porter
#: is pure and deterministic, so every answer after the first is free.
#:
#: Unbounded on purpose, and safe to be: the key space is the distinct words of
#: a fixed corpus on disk, not anything a caller can grow. The TypeScript twin
#: bounds its own cache, because there the input is a stranger's paste.
_CACHE = {}


def stem(word):
    """`screens` and `screening` -> `screen`. Lowercase in, lowercase out."""
    cached = _CACHE.get(word)
    if cached is not None:
        return cached
    _CACHE[word] = result = _stem(word)
    return result


def _stem(word):
    if len(word) < MIN_STEM_LENGTH:
        return word

    # ── 1a: plurals ──────────────────────────────────────────────────────────
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # ── 1b: past tense and gerunds ───────────────────────────────────────────
    stripped = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed") and _has_vowel(word[:-2]):
        word = word[:-2]
        stripped = True
    elif word.endswith("ing") and _has_vowel(word[:-3]):
        word = word[:-3]
        stripped = True

    if stripped:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _double_consonant(word) and word[-1] not in "lsz":
            word = word[:-1]
        elif _measure(word) == 1 and _cvc(word):
            word += "e"

    # ── 1c: terminal y ───────────────────────────────────────────────────────
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    word = _replace(word, _STEP2, 0)
    word = _replace(word, _STEP3, 0)

    # ── 4: the suffix comes off entirely, and `ion` needs a stem to hang on ──
    if word.endswith("ion"):
        stem_ = word[:-3]
        if _measure(stem_) > 1 and stem_.endswith(("s", "t")):
            word = stem_
        else:
            word = _replace(word, _STEP4, 1)
    else:
        word = _replace(word, _STEP4, 1)

    # ── 5a: terminal e ───────────────────────────────────────────────────────
    if word.endswith("e"):
        without = word[:-1]
        m = _measure(without)
        if m > 1 or (m == 1 and not _cvc(without)):
            word = without

    # ── 5b: doubled l ────────────────────────────────────────────────────────
    if _measure(word) > 1 and _double_consonant(word) and word.endswith("l"):
        word = word[:-1]

    return word


class Surfaces:
    """Stems counted, surface words reported.

    Counting in stem space is the whole point — `screens` and `screening` have
    to land in one bucket — but a report that says a server's tightest cluster
    is `at` when every tool wrote `ATS` has made itself harder to check than
    the thing it is checking. So every stem carries the surface form readers
    will recognise.

    The shortest surface form wins, alphabetical on a tie. Deterministic, which
    matters because these strings end up in committed JSON, and in practice the
    shortest form is the one closest to the lemma: `screen` over `screening`,
    `ats` over nothing else at all.
    """

    def __init__(self):
        self._best = {}

    def add(self, word):
        """Record `word` as a surface form and return its stem."""
        key = stem(word)
        current = self._best.get(key)
        if current is None or (len(word), word) < (len(current), current):
            self._best[key] = word
        return key

    def of(self, key):
        return self._best.get(key, key)

    def all(self, keys):
        return [self.of(key) for key in keys]


#: The process-wide surface registry.
#:
#: A global rather than something threaded through every function, because the
#: study is a single-shot script and the registry is order-independent by
#: construction: shortest form wins, alphabetical on a tie, so the same corpus
#: produces the same strings whichever order the servers are read in.
SURFACES = Surfaces()


def record(word):
    """Stem `word`, remembering it as a surface form. Returns the stem."""
    return SURFACES.add(word)


def surface(key):
    """The surface form to print for a stem."""
    return SURFACES.of(key)


def surfaces(keys):
    return SURFACES.all(keys)
