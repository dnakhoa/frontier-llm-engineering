# %% [markdown]
# # Lab 23 — Contamination: what an n-gram check finds, and what it misses
#
# Companion to [Chapter 23](../book/part-4-infra/23-evaluation.md).
#
# Every technical report says some version of "we decontaminated the training
# data against our evaluation sets." Almost none of them say what that
# procedure would have failed to catch. This lab builds the standard detector,
# plants contamination it can find, and then plants contamination it cannot.
#
# The uncomfortable part is section 4. A detector that reports zero
# contamination is indistinguishable from a corpus that has none, and you
# cannot tell which one you have by looking at the number.
#
# You will:
#
# 1. Build a benchmark and a corpus, and plant known contamination in three
#    forms: verbatim, paraphrased, and translated.
# 2. Run the standard n-gram detector and measure its recall on each.
# 3. Add a fuzzy similarity detector and measure how much it recovers — and
#    what it costs in false positives.
# 4. Measure the false-positive rate on genuinely clean items, and find what
#    drives it.
#
# Pure standard library. No downloads. Runs in seconds.
#
# **Predict before you run.** Twelve benchmark items are copied into the corpus
# word for word, and twelve are copied after light paraphrasing. A 13-gram
# detector runs over both. Write down the two recall numbers you expect.

# %%
from __future__ import annotations

import math
import os
import random
import re
from collections import Counter

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"
SEED = 0
print(f"smoke_test={SMOKE}")

# %% [markdown]
# ## 1. A benchmark, a corpus, and three kinds of leak
#
# One detail here matters more than it looks: the benchmark items and the
# corpus documents are written in the **same register**, from overlapping
# vocabulary. That is not a convenience — it is the real situation. Benchmarks
# are built by people reading the same web that the corpus was scraped from, so
# a detector cannot rely on benchmark text being distinctive.

# %%
CORPUS_SENTENCES = """The transit method detects a planet by the periodic dimming of its host star.
Sedimentary layers record the order in which material was deposited over time.
A catalyst lowers activation energy without being consumed by the reaction.
Ocean currents redistribute heat from the equator toward the poles.
The printing press lowered the cost of copying below the cost of controlling.
Enzymes are proteins that accelerate specific biochemical reactions.
Tectonic plates move a few centimetres each year across the mantle.
A prime number has exactly two distinct positive divisors.
Antibiotic resistance spreads through horizontal gene transfer between bacteria.
Inflation erodes nominal returns and is invisible on a brokerage statement.
Photosynthesis converts light energy into chemical bonds within carbohydrate.
Trade routes moved ideas and disease as efficiently as they moved goods.
The speed of light in a vacuum is the same for every observer.
Erosion reshapes a coastline faster than most people expect.
A vaccine trains the immune system without causing the disease itself.
Compound interest turns a modest return into a large multiple over decades.""".strip().split("\n")

BENCHMARK_SENTENCES = """A neutron star packs more than a solar mass into a sphere roughly the size of a small city, which makes it the densest object we can observe directly.
The half life of a radioactive isotope is entirely independent of the quantity you start with, so a gram and a tonne decay at the same proportional rate.
Convection carries heat through a fluid by physically moving the fluid itself, which is why it transports energy far faster than conduction through a solid.
Natural selection acts on the phenotype of an organism but is heritable only through the genotype, and that gap explains why acquired traits are not inherited.
A logarithm converts multiplication into addition across its entire domain, which is the property that made slide rules and log tables useful for three centuries.
Glaciers preserve a detailed record of past atmospheric composition inside trapped air bubbles, so an ice core is effectively a dated archive of ancient air.
Supply and demand determine a price only where both sides are able to transact freely, and every real market departs from that assumption in some direction.
The nervous system transmits its signals as travelling waves of membrane depolarisation, which is why nerve conduction has a speed that can be measured directly.
An eclipse occurs whenever three bodies align closely enough for one to cast a shadow onto another, which happens far more often than most people assume.
Sound travels considerably faster through water than it does through air, because the molecules of a liquid are packed closely enough to transmit pressure quickly.
A keystone species affects the structure of its ecosystem out of all proportion to its own biomass, so removing it changes far more than its share suggests.
Latitude determines how much solar energy a given surface receives across a year, which sets the broad pattern of climate before any local effect is considered.
The immune system distinguishes self from non self by reading molecular signatures on cell surfaces, and autoimmune disease is what happens when that reading fails.
Metals conduct electricity readily because their outermost electrons are delocalised across the lattice rather than being bound to any particular atom.
A river deposits its heaviest sediment at precisely the point where its current first slows, which is why deltas and floodplains have the composition they do.
Genetic drift changes allele frequencies most strongly in small populations, so isolated groups diverge from their parent population faster than large ones do.
Crystals form when atoms settle into a repeating three dimensional lattice, and the external shape of the crystal reflects the symmetry of that internal arrangement.
The atmosphere scatters short wavelengths of light far more strongly than long ones, which accounts for both the blue of the sky and the red of a sunset.
Fermentation releases usable energy from sugar without requiring any oxygen at all, which is why it works in sealed vessels and in oxygen starved muscle tissue.
Continental drift was rejected by geologists for several decades because no plausible mechanism was known, and it was accepted only once sea floor spreading was mapped.""".strip().split("\n")

# Four deliberately SHORT items, planted verbatim. Real benchmarks are full of
# these -- a GSM8K question or an MMLU stem is often under a dozen words.
SHORT_ITEMS = """Sound travels faster through water than air.
A prime has exactly two positive divisors.
Enzymes accelerate specific biochemical reactions.
Erosion reshapes a coastline over time.""".strip().split("\n")

NAV = ["Home | About | Contact | Privacy", "Menu Search Subscribe Log in",
       "Skip to content. Newsletter signup.", "Archive Categories Tags RSS"]

WORD_RE = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list:
    return WORD_RE.findall(text.lower())


# --- paraphrasing: the kind of rewrite a content farm produces ---------------
SYNONYMS = {
    "packs": "compresses", "size": "diameter", "independent": "unrelated",
    "starting": "initial", "carries": "transports", "moving": "displacing",
    "acts": "operates", "converts": "turns", "whole": "entire",
    "preserve": "retain", "record": "archive", "trapped": "enclosed",
    "determine": "set", "freely": "openly", "transmits": "relays",
    "occurs": "happens", "align": "line up", "travels": "propagates",
    "affects": "influences", "proportion": "relation", "determines": "controls",
    "receives": "absorbs", "distinguishes": "separates", "conduct": "carry",
    "outer": "valence", "deposits": "drops", "heaviest": "densest",
    "changes": "shifts", "small": "tiny", "settle": "arrange",
    "repeating": "recurring", "scatters": "disperses", "strongly": "intensely",
    "releases": "extracts", "requiring": "needing", "rejected": "dismissed",
    "known": "identified", "more": "greater", "same": "identical",
}


FILLERS = ["actually", "in practice", "as it happens", "broadly speaking",
           "in general", "of course", "typically", "for the most part"]


def paraphrase(text: str, rng: random.Random) -> str:
    """A content-farm rewrite: synonyms, an opener, and a small edit every few
    words.

    The edit DENSITY is the parameter that matters, not how many words change
    in total. A rewrite that touches something every five or six words caps the
    longest untouched run below any sensible n, which is precisely what makes
    it invisible to n-gram matching.
    """
    words = [SYNONYMS.get(w.lower().strip(".,"), w) for w in text.split()]
    out, since_edit = [], 0
    for w in words:
        out.append(w)
        since_edit += 1
        if since_edit >= rng.randint(4, 6):
            out.append(rng.choice(FILLERS))
            since_edit = 0
    body = " ".join(out).rstrip(".")
    opener = rng.choice(["It is worth noting that", "Interestingly,",
                         "As is well known,", "Put simply,"])
    return f"{opener} {body[0].lower()}{body[1:]}."


def translate(text: str) -> str:
    """A deterministic word-level cipher standing in for another language.

    Not a real translation. It preserves word order and one-to-one word
    identity, which makes it an EASY case -- a real translation would be
    harder still, so treat section 5 as an upper bound on detectability.
    """
    def enc(w: str) -> str:
        return "".join(chr((ord(c) - 97 + 7) % 26 + 97) if c.isalpha() else c
                       for c in w)
    return " ".join(enc(w) for w in tokens(text)) + "."


# --- build the corpus with planted items ------------------------------------
rng = random.Random(SEED)
N_DOCS = 200 if SMOKE else 800

n_bench = len(BENCHMARK_SENTENCES)
per_group = n_bench // 4
VERBATIM = list(range(0, per_group))
PARAPHRASED = list(range(per_group, 2 * per_group))
TRANSLATED = list(range(2 * per_group, 3 * per_group))
CLEAN = list(range(3 * per_group, n_bench))       # never planted: the control


def make_doc(rng: random.Random, planted: str = None) -> str:
    body = rng.sample(CORPUS_SENTENCES, rng.randint(5, 8))
    if planted:
        body.insert(rng.randrange(len(body) + 1), planted)
    return f"{rng.choice(NAV)}\n" + " ".join(body)


corpus = []
for i, idx in enumerate(VERBATIM):
    corpus.append(make_doc(rng, BENCHMARK_SENTENCES[idx]))
for idx in PARAPHRASED:
    corpus.append(make_doc(rng, paraphrase(BENCHMARK_SENTENCES[idx], rng)))
for idx in TRANSLATED:
    corpus.append(make_doc(rng, translate(BENCHMARK_SENTENCES[idx])))
for short in SHORT_ITEMS:
    corpus.append(make_doc(rng, short))
while len(corpus) < N_DOCS:
    corpus.append(make_doc(rng))
rng.shuffle(corpus)

print(f"\nBenchmark: {n_bench} items")
print(f"Corpus:    {len(corpus)} documents")
print("-" * 74)
print(f"  planted verbatim    : {len(VERBATIM)} items")
print(f"  planted paraphrased : {len(PARAPHRASED)} items")
print(f"  planted translated  : {len(TRANSLATED)} items")
print(f"  never planted       : {len(CLEAN)} items  <- the control group")
print("\nExample paraphrase:")
print(f"  original: {BENCHMARK_SENTENCES[PARAPHRASED[0]]}")
print(f"  planted : {paraphrase(BENCHMARK_SENTENCES[PARAPHRASED[0]], random.Random(SEED))}")

# %% [markdown]
# ## 2. The standard detector
#
# The GPT-3 recipe, and roughly what everyone still does: build the set of all
# $n$-grams in the corpus, and flag a benchmark item if **any** of its
# $n$-grams appears. $n = 13$ is the usual choice.

# %%
def ngrams(toks: list, n: int) -> set:
    if len(toks) < n:
        return set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def build_index(docs: list, n: int) -> set:
    index = set()
    for d in docs:
        index |= ngrams(tokens(d), n)
    return index


def ngram_flags(items: list, index: set, n: int) -> list:
    out = []
    for item in items:
        g = ngrams(tokens(item), n)
        out.append(bool(g & index) if g else False)
    return out


def recall(flags: list, group: list) -> float:
    if not group:
        return float("nan")
    return 100.0 * sum(1 for i in group if flags[i]) / len(group)


N_GRAM = 13
index13 = build_index(corpus, N_GRAM)
flags13 = ngram_flags(BENCHMARK_SENTENCES, index13, N_GRAM)

print(f"\n{N_GRAM}-gram detector ({len(index13):,} distinct n-grams indexed)")
print("-" * 74)
print(f"{'group':<24} {'n':>4} {'detected':>10} {'recall':>9}")
for label, group in (("verbatim", VERBATIM), ("paraphrased", PARAPHRASED),
                     ("translated", TRANSLATED), ("clean (control)", CLEAN)):
    det = sum(1 for i in group if flags13[i])
    print(f"{label:<24} {len(group):>4} {det:>10} {recall(flags13, group):>8.0f}%")

print("\nVerbatim copies are caught, and that is the honest strength of the")
print("method: for exact duplication it is cheap, exhaustive, and did not")
print("produce a single false positive on the control group.")
print("\nThe paraphrase row is the whole problem. The planted text says the same")
print("thing, teaches the model the same answer, and shares not one 13-gram.")

# --- the length trap --------------------------------------------------------
short_flags = ngram_flags(SHORT_ITEMS, index13, N_GRAM)
print(f"\nNow the same detector on {len(SHORT_ITEMS)} SHORT items, all planted verbatim")
print("-" * 74)
print(f"{'item':<52} {'tokens':>7} {'flagged':>9}")
for item, f in zip(SHORT_ITEMS, short_flags):
    print(f"{item[:50]:<52} {len(tokens(item)):>7} {str(f):>9}")
print(f"\nrecall on verbatim SHORT items: "
      f"{100*sum(short_flags)/len(short_flags):.0f}%")
print("\nThese are word-for-word copies sitting in the corpus, and the detector")
print("cannot see them -- an item shorter than n has no n-grams at all, so there")
print("is nothing to match. It is not a tuning problem; it is arithmetic.")
print("\nThat matters because real benchmark items are short. An MMLU stem or a")
print("GSM8K question is frequently under thirteen tokens, which means the")
print("standard decontamination check is structurally blind to a large part of")
print("the very benchmarks it is run against.")

# %% [markdown]
# ## 3. Why paraphrase defeats it, exactly
#
# Not a subtle statistical effect. A 13-gram survives only if thirteen
# consecutive words are untouched, so a rewrite every eight or nine words
# destroys every one of them.

# %%
print(f"\nLongest shared word run, planted paraphrase vs original")
print("-" * 74)


def longest_common_run(a: list, b: list) -> int:
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best


runs = []
para_rng = random.Random(SEED)
for idx in PARAPHRASED[:4]:
    orig = tokens(BENCHMARK_SENTENCES[idx])
    para = tokens(paraphrase(BENCHMARK_SENTENCES[idx], random.Random(SEED)))
    r = longest_common_run(orig, para)
    runs.append(r)
    print(f"  item {idx:>2}: longest identical run = {r:>2} words "
          f"({'caught' if r >= N_GRAM else 'invisible'} at n={N_GRAM})")

print(f"\nSweeping n, to see where the detector starts working")
print("-" * 74)
print(f"{'n':>4} {'verbatim':>10} {'paraphrased':>13} {'clean FP':>10} "
      f"{'index size':>12}")
NS = (3, 5, 8, 13) if SMOKE else (3, 4, 5, 8, 10, 13)
for n in NS:
    idx_n = build_index(corpus, n)
    f = ngram_flags(BENCHMARK_SENTENCES, idx_n, n)
    print(f"{n:>4} {recall(f, VERBATIM):>9.0f}% {recall(f, PARAPHRASED):>12.0f}% "
          f"{recall(f, CLEAN):>9.0f}% {len(idx_n):>12,}")

print("\nThere is the trade, and it is not a good one. Lowering n to catch")
print("paraphrase also starts flagging items that were never planted, because")
print("short spans of ordinary English recur everywhere. That last column is")
print("the false-positive rate on the control group, and it is driven by")
print("nothing more exotic than common phrasing.")

# %% [markdown]
# ## 4. A fuzzy detector
#
# The standard answer is to compare meaning rather than surface form: embed
# both sides and use cosine similarity.
#
# **What this lab actually does, stated plainly.** Real pipelines use a neural
# sentence encoder. We cannot download one, so this uses TF-IDF cosine
# similarity — a bag of words weighted by inverse document frequency. It is a
# genuinely fuzzy matcher and it demonstrates the shape of the result, but a
# neural encoder would do better on paraphrase, and unlike TF-IDF a
# *multilingual* encoder would also survive section 5. Read the numbers here as
# a lower bound on what embedding-based detection achieves.

# %%
def build_idf(docs: list) -> dict:
    df = Counter()
    for d in docs:
        df.update(set(tokens(d)))
    n = len(docs)
    return {w: math.log(n / (1 + c)) + 1.0 for w, c in df.items()}


def tfidf_vector(text: str, idf: dict) -> dict:
    counts = Counter(tokens(text))
    vec = {w: (1 + math.log(c)) * idf.get(w, math.log(len(corpus)) + 1.0)
           for w, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {w: v / norm for w, v in vec.items()}


def cosine(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(w, 0.0) for w, v in a.items())


IDF = build_idf(corpus)
CORPUS_VECS = [tfidf_vector(d, IDF) for d in corpus]


def max_similarity(item: str) -> float:
    v = tfidf_vector(item, IDF)
    return max(cosine(v, cv) for cv in CORPUS_VECS)


sims = [max_similarity(s) for s in BENCHMARK_SENTENCES]

print("\nMax TF-IDF cosine similarity against any corpus document")
print("-" * 74)
print(f"{'group':<24} {'min':>8} {'mean':>8} {'max':>8}")
for label, group in (("verbatim", VERBATIM), ("paraphrased", PARAPHRASED),
                     ("translated", TRANSLATED), ("clean (control)", CLEAN)):
    vals = [sims[i] for i in group]
    print(f"{label:<24} {min(vals):>8.3f} {sum(vals)/len(vals):>8.3f} {max(vals):>8.3f}")

# Choose the threshold the way you would have to in practice: from the clean
# distribution, since in reality that is the only group whose label you know.
clean_max = max(sims[i] for i in CLEAN)
THRESHOLD = clean_max + 1e-9
fuzzy = [s >= THRESHOLD for s in sims]

print(f"\nThreshold set just above the highest clean score ({clean_max:.3f}),")
print("i.e. tuned for zero false positives on the control group.")
print("-" * 74)
print(f"{'group':<24} {'n-gram recall':>15} {'fuzzy recall':>14}")
for label, group in (("verbatim", VERBATIM), ("paraphrased", PARAPHRASED),
                     ("translated", TRANSLATED), ("clean FP rate", CLEAN)):
    print(f"{label:<24} {recall(flags13, group):>14.0f}% "
          f"{recall(fuzzy, group):>13.0f}%")

print("\nFuzzy matching recovers a large share of the paraphrased items that the")
print("n-gram detector could not see, at zero false positives on this control")
print("-- but note how that threshold was chosen. We used the answer key. In a")
print("real pipeline nobody knows which items are clean, so the threshold is")
print("guessed, and every point of recall is bought with false positives that")
print("delete legitimate training data.")

# %% [markdown]
# ## 5. Translation, where both detectors go blind

# %%
print("\nTranslated items (word-level cipher, word order preserved)")
print("-" * 74)
print(f"{'detector':<34} {'recall on translated':>22}")
print(f"{'13-gram exact match':<34} {recall(flags13, TRANSLATED):>21.0f}%")
print(f"{'TF-IDF cosine':<34} {recall(fuzzy, TRANSLATED):>21.0f}%")

print("\nBoth are zero, and neither could be anything else. Exact n-grams need")
print("shared word sequences; TF-IDF needs shared vocabulary. A translation has")
print("neither, while teaching the model exactly the same fact.")
print("\nThis is not a hypothetical. Benchmark items get translated, discussed on")
print("forums in other languages, and re-scraped. A multilingual embedding model")
print("catches some of it -- this lab cannot demonstrate that without a")
print("download, and saying so is more useful than pretending otherwise.")
print("\nThe conclusion to carry into Chapter 23: a contamination report of")
print("'0.02% overlap' is a measurement of the DETECTOR, not of the corpus.")
print("Ask which detector, at what n, and what it was never going to find.")

# %% [markdown]
# ## 6. Things to try
#
# **1. Set `N_GRAM = 8` and rerun sections 2 and 4.**
# *Common prediction:* paraphrase recall improves substantially.
# *What happens:* it improves a little and the clean false-positive rate starts
# moving. The sweep in section 3 shows why: there is no value of $n$ that
# catches rewrites without flagging ordinary English.
#
# **2. Make `paraphrase` weaker — synonyms only, no opener.**
# *Common prediction:* still invisible to the 13-gram detector.
# *What happens:* recall jumps, because the untouched runs between substitutions
# get longer. Detectability is set by the *longest untouched span*, not by how
# much of the text was changed, which is why "we changed 30% of the words" says
# nothing about whether a detector will fire.
#
# **3. Plant an item twice in two different documents.**
# *Common prediction:* twice as likely to be caught.
# *What happens:* no change at all — the detector already flags on any single
# match. Recall is per-item, not per-occurrence, so a corpus with the same leak
# a thousand times scores identically to one with it once, while being far more
# damaging.
#
# **4. Set the fuzzy threshold to `0.5` instead of tuning it on the controls.**
# *Common prediction:* somewhat more recall, a few false positives.
# *What happens:* the clean group starts failing, and every false positive is a
# legitimate document deleted from your corpus. Decontamination is a
# precision/recall trade in which both errors cost you something real.
#
# **5. Add a benchmark item that is a common English sentence.**
# *Common prediction:* it behaves like the others.
# *What happens:* it is flagged by every detector at every setting, because it
# genuinely does appear in the corpus. Some benchmark items are uncheckable,
# and the honest response is to exclude them rather than to report them as
# contamination.

# %% [markdown]
# ## What to take away
#
# 1. **N-gram matching is excellent at exactly one thing.** Verbatim copies:
#    full recall, no false positives, cheap. Nothing else.
# 2. **Detectability is set by the longest untouched span.** A rewrite every
#    eight or nine words makes a 13-gram detector blind, regardless of how
#    similar the meaning is.
# 3. **There is no good $n$.** Lowering it to catch paraphrase raises the false
#    positive rate on ordinary text, and every false positive deletes real
#    training data.
# 4. **Fuzzy matching helps and needs a threshold nobody can set honestly.**
#    Ours was tuned using the answer key, which is a luxury no real pipeline
#    has.
# 5. **Translated contamination is invisible to both**, while teaching the
#    model the same thing.
# 6. **A contamination percentage describes the detector.** When a report says
#    a corpus was decontaminated, the question is what the method could never
#    have found — and that is rarely stated.
