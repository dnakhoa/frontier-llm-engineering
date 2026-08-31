# %% [markdown]
# # Lab 3 — Dedup and quality filtering, the two filters that shape a corpus
#
# Companion to [Chapter 3](../book/part-2-pretraining/03-data-pipelines.md).
#
# Every frontier corpus is built by throwing most of the internet away. Two
# filters do the bulk of that work: **deduplication** and **quality
# classification**.
#
# **Where I went in wrong.** I assumed dedup was the boring filter and quality
# was the interesting one. It is the other way around. Dedup is a genuinely
# subtle algorithm with a threshold you never set directly and cannot easily
# observe. The quality classifier is a fifty-line logistic regression — and it
# took me four rewrites to stop accidentally measuring the wrong thing with it,
# at which point it told me something different from what I had gone in to
# prove. The wrong turns are left in the comments, because they were the part I
# learned from.
#
# You will:
#
# 1. Build a corpus with a *known* duplicate structure, so every claim here is
#    checkable against ground truth rather than asserted.
# 2. Run exact-hash dedup and watch it catch almost nothing.
# 3. Implement MinHash from scratch and watch estimation error fall as
#    $1/\sqrt{k}$ — slowly.
# 4. Implement LSH banding and derive the S-curve that sets your threshold.
# 5. Train a quality classifier and find out what it is actually classifying.
# 6. Reorder the pipeline and measure whether the two orderings agree.
#
# Pure standard library. No downloads, no torch, no GPU. Runs in seconds.
#
# **Predict before you run.** Section 2 asks what fraction of a corpus where
# ~40% of documents live in a duplicate cluster exact-hash dedup will remove.
# Write a number down now.

# %%
from __future__ import annotations

import math
import os
import random
import re
import zlib
from collections import Counter
from dataclasses import dataclass

SMOKE = os.environ.get("FLE_SMOKE_TEST") == "1"
SEED = 0

print(f"smoke_test={SMOKE}")

# %% [markdown]
# ## 1. A corpus with a known duplicate structure
#
# Real crawl dumps do not ship duplicate labels, which makes it impossible to
# tell whether your dedup worked. So we generate a corpus where the answer is
# known: topical sentence pools, wrapped in the navigation chrome real scraped
# pages carry, duplicated in the four ways the web actually duplicates things —
# verbatim syndication, the same article under different chrome, a lightly
# edited rewrite, and a mirror that stacks even more boilerplate on top.
#
# **Honest limitation, stated up front.** This corpus is generated, not
# crawled. The MinHash, the LSH, the classifier and the pipeline comparison are
# all real implementations doing real work, but the *inputs* are synthetic and
# a synthetic corpus is cleaner than Common Crawl in ways that flatter every
# method here. The mechanisms are the lesson; the absolute numbers are an
# upper bound.

# %%
DOMAINS = {
    "astronomy": """A red dwarf burns hydrogen so slowly that its lifetime exceeds the age of the universe.
Parallax gives distances to nearby stars without assuming anything about their brightness.
The cosmic microwave background is the oldest light that can still be detected today.
Spectral lines shift toward the red end of the spectrum when a source recedes from us.
A supernova remnant seeds the surrounding medium with elements heavier than iron.
Gravitational lensing bends light around a mass and produces multiple images of it.
The habitable zone is defined by the range of orbits where liquid water can persist.
Radio telescopes can be combined across continents to sharpen the angular resolution.
Planetary orbits precess slowly under the gravitational influence of neighbouring bodies.
Stellar nurseries begin to collapse once a cloud exceeds its own Jeans mass.
Neutron stars pack more than a solar mass into a sphere the size of a small city.
The interstellar medium is not empty but filled with sparse gas and cold dust.""",
    "cooking": """Browning meat before braising builds flavour through the Maillard reaction.
Resting a roast lets the juices redistribute instead of running out onto the board.
Salt draws moisture out of vegetables and concentrates the flavour of what remains.
A stable emulsion needs an emulsifier, steady agitation, and patience in that order.
Bread dough develops gluten through folding rather than through kneading alone.
Acid brightens a dish that tastes flat even when it has been properly salted.
Blanching sets the colour of green vegetables and halts enzymatic browning.
Stock made from roasted bones has noticeably more body than stock from raw ones.
Cold butter cut into flour is what produces flaky rather than crumbly pastry.
Reducing a sauce concentrates flavour but also concentrates salt, so season it last.
Searing does not seal in moisture, and that claim has been tested many times.
Resting dough in the cold slows fermentation and deepens the finished flavour.""",
    "finance": """A bond price moves inversely to the yield that the market demands for it.
Diversification reduces idiosyncratic risk but it cannot remove market risk.
Compounding turns a modest annual return into a large multiple over several decades.
Liquidity is the ability to sell an asset without moving the price against yourself.
An inverted yield curve has historically preceded most recorded recessions.
Leverage magnifies gains and losses by exactly the same multiplicative factor.
Index funds win on fees, which turns out to be most of what they need to win on.
A margin call forces a sale at precisely the worst possible moment to be selling.
Inflation erodes nominal returns and is invisible on an ordinary brokerage statement.
Correlations tend toward one during a crisis, which is when you need them not to.
Survivorship bias makes historical fund performance look far better than it was.
A stop loss is an instruction that becomes a market order at the worst time.""",
    "biology": """Enzymes lower the activation energy of a reaction without being consumed by it.
Mitochondria carry their own small genome, inherited through the maternal line.
Natural selection acts on phenotype but is only heritable through the genotype.
A cell membrane is selectively permeable and maintains a steep ion gradient.
Antibiotic resistance spreads through horizontal gene transfer between species.
Photosynthesis converts light energy into chemical bonds within carbohydrate.
DNA replication is semi conservative, keeping one strand from the parent molecule.
Apoptosis is programmed cell death and is essential to normal development.
Homeostasis is maintained by negative feedback loops across many organ systems.
A keystone species affects its ecosystem out of all proportion to its own mass.
Ribosomes translate messenger RNA into a chain of amino acids in strict order.
Convergent evolution produces similar forms from entirely unrelated lineages.""",
    "history": """Trade routes moved ideas and disease just as efficiently as they moved goods.
Written law made power legible and therefore contestable in an entirely new way.
The printing press lowered the cost of copying below the cost of controlling.
Empires typically overextend their supply lines well before they lose a battle.
Agricultural surplus is the precondition for any specialised urban labour force.
Plague repeatedly reshaped labour markets by making the survivors comparatively scarce.
Standing armies changed the relationship between a ruler and their own nobility.
Cartography served navigation and conquest long before it ever served curiosity.
Currency debasement is the oldest recorded method of hidden taxation anywhere.
Archives survive by accident far more often than they ever survive by intent.
Census records were built for taxation and conscription rather than for history.
Roads outlast the states that build them and quietly shape what comes next.""",
}

# Each value above is one string; split it into a list of sentences. Leaving this
# out is the single most productive bug I hit writing this lab: rng.sample() on a
# str samples CHARACTERS, so every "document" was a handful of stray letters. The
# pipeline ran, printed plausible tables, and every number in them was noise.
DOMAINS = {k: [ln.strip() for ln in v.strip().split("\n") if ln.strip()]
           for k, v in DOMAINS.items()}

NAV_CHROME = [
    ("Home | About | Contact | Privacy Policy | Terms of Service",
     "Copyright 2024. All rights reserved. Powered by our platform."),
    ("Skip to main content. Menu. Search. Subscribe. Log in.",
     "Share this post on social media. Leave a comment below."),
    ("MENU HOME ARCHIVE CATEGORIES TAGS RSS FEED",
     "Related articles you may also like. Newsletter signup form."),
    ("Navigation: Front page > Section > Article > Print view",
     "This site uses cookies. Accept all. Manage your preferences."),
]

SPAM_TEMPLATES = [
    "BUY {kw} NOW!!! Best {kw} deals online. Cheap {kw}, discount {kw}, {kw} sale. "
    "Click here for {kw}. Top 10 {kw} 2024. {kw} near me. Free shipping on all {kw}.",
    "{kw} {kw} {kw} reviews ratings prices compare. Order {kw} today 50% OFF!!! "
    "Visit our store for {kw}. Call 1-800-555-0199 now. Limited time {kw} offer!!!",
    "Looking for {kw}? We have the best {kw} on the internet. Our {kw} experts all "
    "recommend {kw}. Read more about {kw}. Subscribe for {kw} updates. BUY NOW!!!",
]

SPAM_KEYWORDS = ["insurance", "hosting", "loans", "supplements", "crypto", "vpn"]

# A scraper mirror keeps the whole article and buries it in ad junk. Short enough
# that the document stays a near-duplicate of the original; ugly enough that a
# quality filter hates it. Getting this length right was bug #2 -- see section 7.
JUNK = ("CLICK HERE NOW!!! Sponsored links. Buy cheap deals online 50% OFF!!! "
        "Advertisement. Top 10 offers 2024. Call 1-800-555-0199 today!!!")

# Raw crawl is not only prose and spam. It is also full of source code, licence
# text, and forum posts. These go into the classifier's NEGATIVE class in
# section 6 for exactly the reason they land there in a real pipeline: they are
# what you get when you scrape the web and do not curate.
RAW_CRAWL_SAMPLES = [
    """function handleSubmit(event) { event.preventDefault(); const data = new FormData(form);
    fetch('/api/v1/submit', { method: 'POST', body: data }).then(res => res.json()); }""",
    """import os, sys, json
    def load_config(path): 
        with open(path) as fh: return json.load(fh)
    if __name__ == '__main__': print(load_config(sys.argv[1]))""",
    """THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY AND
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.""",
    """WHEREAS the Party of the First Part hereinafter referred to as the Vendor and
    WHEREAS the Party of the Second Part hereinafter the Purchaser agree pursuant to
    clause 7(b) subsection (iv) as amended thereunder and notwithstanding the foregoing.""",
    """re: re: re: anyone else getting this error??? +1 same here. bump. any fix yet?
    edit: nvm fixed it, was a typo lol. thanks!! ^this. deleted user. [removed]""",
    """<div class="wrapper"><span id="x">&nbsp;</span></div> var _gaq = _gaq || [];
    document.write('<script src="//cdn.example.com/t.js"></script>'); 404 Not Found nginx""",
]


@dataclass
class Doc:
    doc_id: int
    text: str
    cluster: int        # ground truth: same cluster == same underlying article
    domain: str
    kind: str           # "article" | "spam" | "mirror" | "crawl"


def _wrap(body: str, chrome: tuple, repeats: int = 1) -> str:
    head, foot = chrome
    return "\n".join([head] * repeats + [body] + [foot] * repeats)


def build_corpus(n_singletons: int, n_clusters: int, seed: int = SEED) -> list:
    rng = random.Random(seed)
    docs: list = []
    cluster_id = 0

    def article(domain: str, n_sent: int) -> list:
        return rng.sample(DOMAINS[domain], n_sent)

    # --- singletons: genuinely distinct documents -------------------------
    for _ in range(n_singletons):
        domain = rng.choice(list(DOMAINS))
        body = " ".join(article(domain, 8))
        docs.append(Doc(0, _wrap(body, rng.choice(NAV_CHROME)), cluster_id, domain, "article"))
        cluster_id += 1

    # --- duplicate clusters ------------------------------------------------
    # The chrome is fixed per cluster so that "exact duplicate" really is
    # byte-identical. Getting this wrong was bug #1 in this lab: randomising
    # the chrome per member meant section 2 had nothing to find.
    for _ in range(n_clusters):
        domain = rng.choice(list(DOMAINS))
        base = article(domain, 8)
        chrome = rng.choice(NAV_CHROME)
        body = " ".join(base)
        docs.append(Doc(0, _wrap(body, chrome), cluster_id, domain, "article"))

        variants = rng.sample(["exact", "template", "edit", "reorder"], rng.randint(1, 3))
        if cluster_id % 2 == 0:
            variants.append("mirror")
        for variant in variants:
            if variant == "exact":
                # Syndicated verbatim: the same bytes on another domain.
                docs.append(Doc(0, _wrap(body, chrome), cluster_id, domain, "article"))
            elif variant == "template":
                # Same article, different site chrome.
                other = rng.choice([c for c in NAV_CHROME if c != chrome])
                docs.append(Doc(0, _wrap(body, other), cluster_id, domain, "article"))
            elif variant == "edit":
                # One sentence swapped for another from the same topic.
                pool = [s for s in DOMAINS[domain] if s not in base]
                edited = list(base)
                if pool:
                    edited[rng.randrange(len(edited))] = rng.choice(pool)
                docs.append(Doc(0, _wrap(" ".join(edited), chrome), cluster_id, domain, "article"))
            elif variant == "reorder":
                shuffled = list(base)
                rng.shuffle(shuffled)
                docs.append(Doc(0, _wrap(" ".join(shuffled), chrome), cluster_id, domain, "article"))
            else:
                # A scraper mirror: the same article with ad junk appended. It
                # stays similar enough to be a duplicate and ugly enough for the
                # quality filter to reject -- which is what makes section 7 work.
                docs.append(Doc(0, _wrap(body + " " + JUNK, chrome), cluster_id, domain, "mirror"))
        cluster_id += 1

    # --- spam and raw crawl: the negative class ---------------------------
    for _ in range(max(6, n_singletons // 3)):
        kw = rng.choice(SPAM_KEYWORDS)
        body = rng.choice(SPAM_TEMPLATES).format(kw=kw)
        docs.append(Doc(0, _wrap(body, rng.choice(NAV_CHROME)), cluster_id, "spam", "spam"))
        cluster_id += 1

    for _ in range(max(6, n_singletons // 4)):
        body = rng.choice(RAW_CRAWL_SAMPLES)
        docs.append(Doc(0, _wrap(body, rng.choice(NAV_CHROME)), cluster_id, "crawl", "crawl"))
        cluster_id += 1

    rng.shuffle(docs)
    for i, d in enumerate(docs):
        d.doc_id = i
    return docs


N_SINGLETONS = 40 if SMOKE else 200
N_CLUSTERS = 15 if SMOKE else 70

CORPUS = build_corpus(N_SINGLETONS, N_CLUSTERS)
sizes = Counter(d.cluster for d in CORPUS)
n_dup_members = sum(v for v in sizes.values() if v > 1)
kinds = Counter(d.kind for d in CORPUS)

print(f"corpus: {len(CORPUS)} documents in {len(sizes)} ground-truth clusters")
print(f"  living in a duplicate cluster: {n_dup_members} "
      f"({100*n_dup_members/len(CORPUS):.0f}%)")
print(f"  by kind: {dict(kinds)}")

# %% [markdown]
# ## 2. Exact-hash dedup, and how little it catches
#
# The cheapest possible dedup: hash the document, keep the first occurrence of
# each hash. One pass, no memory to speak of. It is also the filter people most
# often assume is sufficient.

# %%
def exact_dedup(docs: list) -> set:
    seen, kept = set(), set()
    for d in docs:
        h = zlib.crc32(d.text.encode())
        if h not in seen:
            seen.add(h)
            kept.add(d.doc_id)
    return kept


kept_exact = exact_dedup(CORPUS)
removed_exact = len(CORPUS) - len(kept_exact)
ideal_removed = len(CORPUS) - len(sizes)   # keep one survivor per cluster

print("Exact-hash deduplication")
print("-" * 74)
print(f"{'documents in':>26}: {len(CORPUS)}")
print(f"{'removed by exact hash':>26}: {removed_exact}")
print(f"{'should be removed':>26}: {ideal_removed}  (one survivor per cluster)")
print(f"{'recall':>26}: {100*removed_exact/max(ideal_removed,1):.1f}%")
print()
print("Exact hashing catches only byte-identical documents. A swapped sentence,")
print("a different cookie banner, a reordered paragraph -- all sail through.")
print("Note this is the CEILING, not a bad run: our generator produces verbatim")
print("copies deliberately, and real syndication is messier than that.")

# %% [markdown]
# ## 3. Shingles and true Jaccard
#
# To catch near-duplicates you need a similarity, not an equality. Represent a
# document as its set of overlapping word $n$-grams (*shingles*) and compare
# sets with Jaccard similarity:
#
# $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
#
# Shingle length is not a sensitivity dial — it is the *definition* of what
# "the same document" means. Too short and every English document overlaps;
# too long and one edited word destroys the match.

# %%
WORD_RE = re.compile(r"[a-z0-9']+")
SHINGLE_N = 5


def shingles(text: str, n: int = SHINGLE_N) -> set:
    words = WORD_RE.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


SHINGLE_SETS = {d.doc_id: shingles(d.text) for d in CORPUS}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


TAU = 0.70          # what we are calling a duplicate, on TRUE Jaccard

ids = sorted(SHINGLE_SETS)
true_dup_pairs = set()
all_pairs = 0
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        all_pairs += 1
        if jaccard(SHINGLE_SETS[ids[i]], SHINGLE_SETS[ids[j]]) >= TAU:
            true_dup_pairs.add((ids[i], ids[j]))

print(f"\n{all_pairs:,} pairs compared exhaustively")
print(f"{len(true_dup_pairs):,} pairs exceed J >= {TAU} -- this is the ground truth")
print("\nThat exhaustive loop is the entire problem. 15T tokens is roughly 10^10")
print("documents, and 10^10 documents is 5x10^19 pairs. Everything below exists")
print("to avoid writing this loop.")

# %% [markdown]
# ## 4. MinHash: estimating Jaccard in constant space
#
# For a random permutation $\pi$ of the shingle universe,
#
# $$\Pr[\min \pi(A) = \min \pi(B)] = J(A, B)$$
#
# Take $k$ independent permutations, record the minimum under each, and the
# fraction of agreeing positions estimates Jaccard. Every document collapses to
# $k$ integers regardless of length.
#
# The standard deviation of that estimate is $\sqrt{J(1-J)/k}$ — error falls as
# $1/\sqrt{k}$, which is *slow*. That is the whole reason $k$ is 128 and not 16.

# %%
MERSENNE = (1 << 61) - 1
MAXHASH = (1 << 32) - 1


def make_permutations(k: int, seed: int = SEED) -> list:
    rng = random.Random(seed)
    return [(rng.randrange(1, MERSENNE), rng.randrange(0, MERSENNE)) for _ in range(k)]


def minhash(shingle_set: set, perms: list) -> tuple:
    if not shingle_set:
        return tuple(MAXHASH for _ in perms)
    base = [zlib.crc32(s.encode()) for s in shingle_set]
    return tuple(min(((a * x + b) % MERSENNE) & MAXHASH for x in base) for a, b in perms)


def estimate(sig_a: tuple, sig_b: tuple) -> float:
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


print("\nMinHash estimation error vs permutation count")
print("-" * 74)
print(f"{'k':>6} {'mean abs error':>16} {'theory sqrt(J(1-J)/k)':>24}")

rng_pairs = random.Random(SEED)
sample_pairs = [tuple(sorted(rng_pairs.sample(ids, 2)))
                for _ in range(150 if SMOKE else 400)]

PERM_COUNTS = (16, 32, 128) if SMOKE else (16, 32, 64, 128, 256)
for k in PERM_COUNTS:
    perms = make_permutations(k)
    sigs = {d: minhash(SHINGLE_SETS[d], perms) for d in ids}
    errs, theory = [], []
    for i, j in sample_pairs:
        tj = jaccard(SHINGLE_SETS[i], SHINGLE_SETS[j])
        errs.append(abs(estimate(sigs[i], sigs[j]) - tj))
        theory.append(math.sqrt(max(tj * (1 - tj), 0.0) / k))
    print(f"{k:>6} {sum(errs)/len(errs):>16.4f} {sum(theory)/len(theory):>24.4f}")

print("\nMeasured error tracks the theory: quadrupling k halves the error. That")
print("is an expensive curve to ride, and it is why halving k to save memory")
print("costs more accuracy than people expect.")

# %% [markdown]
# ## 5. LSH banding, and the S-curve that sets your threshold
#
# Comparing every signature pair is still $O(n^2)$. LSH fixes that: split each
# $k$-element signature into $b$ bands of $r$ rows ($k = br$) and hash each
# band. Two documents are *candidates* if any band matches exactly.
#
# The probability a pair of similarity $J$ becomes a candidate is
#
# $$P(J) = 1 - (1 - J^{\,r})^{b}$$
#
# an S-curve with its knee near $(1/b)^{1/r}$.
#
# **This is the thing I had backwards.** I thought of the dedup threshold as a
# number you configure. You do not. You choose $b$ and $r$, and the threshold
# is whatever those imply. Every LSH misconfiguration I have since seen is
# someone setting a similarity cutoff in one place and a banding in another and
# never noticing they disagree.

# %%
def lsh_candidates(sigs: dict, bands: int, rows: int) -> set:
    buckets: dict = {}
    for doc_id, sig in sigs.items():
        for band in range(bands):
            buckets.setdefault((band, sig[band * rows:(band + 1) * rows]), []).append(doc_id)
    pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        members.sort()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


def prf(candidates: set, truth: set) -> tuple:
    tp = len(candidates & truth)
    return (tp / len(candidates) if candidates else 0.0,
            tp / len(truth) if truth else 0.0)


CONFIGS = [
    (128, 16, 8),      # the common default
    (128, 32, 4),      # exercise 3.7 item 2
    (128, 8, 16),      # stricter
    (32, 4, 8),        # exercise 3.7 item 1: fewer permutations
]

print("\nLSH banding: what (b, r) actually buys you")
print("-" * 74)
print(f"{'k':>5} {'b':>4} {'r':>4} {'threshold':>11} {'candidates':>12} "
      f"{'precision':>11} {'recall':>9}")
for k, b, r in CONFIGS:
    perms = make_permutations(k)
    sigs = {d: minhash(SHINGLE_SETS[d], perms) for d in ids}
    cands = lsh_candidates(sigs, b, r)
    p, rec = prf(cands, true_dup_pairs)
    print(f"{k:>5} {b:>4} {r:>4} {(1.0/b)**(1.0/r):>11.3f} {len(cands):>12,} "
          f"{p*100:>10.1f}% {rec*100:>8.1f}%")

print("\nP(candidate) as a function of true Jaccard")
print("-" * 74)
print(f"{'J':>6} " + " ".join(f"{f'b={b},r={r}':>12}" for _, b, r in CONFIGS[:3]))
for jv in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    cells = " ".join(f"{(1 - (1 - jv ** r) ** b)*100:>11.1f}%" for _, b, r in CONFIGS[:3])
    print(f"{jv:>6.2f}  {cells}")

print("\n(16, 8) -> (32, 4) drops the knee from ~0.71 to ~0.42: more candidates,")
print("higher recall, worse precision. Precision matters less than it looks --")
print("production pipelines verify candidates exactly before dropping them, so a")
print("false candidate costs compute. A MISSED duplicate costs corpus, forever,")
print("and you never find out. Bias toward recall.")

# %% [markdown]
# ## 6. A quality classifier, and what it is really classifying
#
# The standard recipe: label a positive class (curated reference prose) and a
# negative class (raw crawl), train a cheap linear classifier on n-gram
# features, keep whatever scores above a threshold.
#
# **This section took four rewrites and changed my mind twice.** Version one used
# cartoon SEO spam as the whole negative class; the classifier scored 92% and
# kept every probe I threw at it, and I nearly shipped a lab whose prose claimed
# the opposite of what it printed. Version two fixed the negative class to look
# like real crawl — code, licence text, forum noise, markup. Version three
# discovered the classifier was really scoring document *length*, because my
# probes were short snippets and every corpus document was eight sentences in
# navigation chrome. Version four controls for that, and only then does the
# actual behaviour show up — which turned out not to be the thing I set out to
# demonstrate. All four versions are visible in the code and comments below.

# %%
STOPWORDS = set("the a an of and or to in is are was were it its for on with that this by as at from be".split())
# 512 buckets, not 4096. With a corpus this small a wide hashed space lets the
# model memorise every document and score 100% held-out while generalising to
# nothing -- bug #3 in this lab. Deliberate collisions force it to lean on the
# structural features below, which is also what a real fastText quality filter
# ends up doing.
N_HASH = 512
EXTRA = 8


def featurize(text: str) -> dict:
    words = WORD_RE.findall(text.lower())
    if not words:
        return {}
    feats: dict = {}
    for w in words:
        idx = zlib.crc32(w.encode()) % N_HASH
        feats[idx] = feats.get(idx, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    for idx in list(feats):
        feats[idx] /= norm

    n = len(words)
    letters = [c for c in text if c.isalpha()]
    feats[N_HASH + 0] = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
    feats[N_HASH + 1] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    feats[N_HASH + 2] = min(sum(len(w) for w in words) / n, 12.0) / 12.0
    feats[N_HASH + 3] = min(text.count("!") / max(len(text), 1) * 100, 1.0)
    feats[N_HASH + 4] = sum(1 for w in words if w in STOPWORDS) / n
    feats[N_HASH + 5] = len(set(words)) / n
    # Symbol density and the fraction of tokens that are ordinary words. Prose
    # scores low and high respectively; code, markup and tables do the reverse.
    feats[N_HASH + 6] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    feats[N_HASH + 7] = sum(1 for w in words if w.isalpha()) / n
    return feats


def train_logreg(samples: list, epochs: int, lr: float = 0.5, l2: float = 2e-3) -> tuple:
    w = [0.0] * (N_HASH + EXTRA)
    bias = 0.0
    rng = random.Random(SEED)
    order = list(range(len(samples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for si in order:
            feats, y = samples[si]
            z = bias + sum(w[i] * v for i, v in feats.items())
            p = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
            g = p - y
            bias -= lr * g
            for i, v in feats.items():
                w[i] -= lr * (g * v + l2 * w[i])
    return w, bias


def score(w: list, bias: float, feats: dict) -> float:
    z = bias + sum(w[i] * v for i, v in feats.items())
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


# Positive = curated article prose. Negative = raw crawl (spam, mirrors, code,
# licence text, forum noise) -- i.e. what you get if you do not curate.
#
# The experiment: hold out an entire DOMAIN. History articles are perfectly good
# prose, identical in register to the training positives, and the classifier
# never sees one. If it has learned "quality", it will keep them. If it has
# learned "these topics", it will not.
HELD_OUT_DOMAIN = "history"

train_docs = [d for d in CORPUS if d.domain != HELD_OUT_DOMAIN]
labelled = [(featurize(d.text), 1.0 if d.kind == "article" else 0.0) for d in train_docs]
split = int(0.75 * len(labelled))
train, held = labelled[:split], labelled[split:]

EPOCHS = 4 if SMOKE else 15
W, B = train_logreg(train, EPOCHS)

correct = sum(1 for f, y in held if (score(W, B, f) >= 0.5) == (y >= 0.5))
print("\nQuality classifier, in-domain")
print("-" * 74)
print(f"held-out-DOCUMENT accuracy: {100*correct/max(len(held),1):.1f}% "
      f"on {len(held)} documents")
print("A number that looks like success and tells you almost nothing.")

# Real pipelines do not threshold at p=0.5. They sort the corpus by score and
# keep a percentile -- because the budget is a token count, not a probability.
# We do the same, and it turns out to matter a great deal.
quality = {d.doc_id: score(W, B, featurize(d.text)) for d in CORPUS}
DROP_FRACTION = 0.30
_ranked = sorted(quality.values())
QUALITY_THRESHOLD = _ranked[int(DROP_FRACTION * len(_ranked))]


def percentile_of(value: float) -> float:
    return 100.0 * sum(1 for v in _ranked if v <= value) / len(_ranked)


print(f"keeping the top {100*(1-DROP_FRACTION):.0f}% by score "
      f"=> cutoff at p={QUALITY_THRESHOLD:.3f}")

print("\nMean score by document kind (ground truth we would not have in reality):")
_by_kind: dict = {}
for _d in CORPUS:
    _by_kind.setdefault(_d.kind, []).append(quality[_d.doc_id])
for _k in sorted(_by_kind):
    _v = _by_kind[_k]
    print(f"  {_k:<10} n={len(_v):>3}  mean={sum(_v)/len(_v):.3f}")

# The experiment proper: same register, same quality, unseen topic.
_seen = [quality[d.doc_id] for d in CORPUS
         if d.kind == "article" and d.domain != HELD_OUT_DOMAIN]
_unseen = [quality[d.doc_id] for d in CORPUS
           if d.kind == "article" and d.domain == HELD_OUT_DOMAIN]
_kept_seen = sum(1 for v in _seen if v >= QUALITY_THRESHOLD)
_kept_unseen = sum(1 for v in _unseen if v >= QUALITY_THRESHOLD)

print(f"\nArticles from the four TRAINING domains: n={len(_seen)}  "
      f"mean={sum(_seen)/len(_seen):.3f}  kept={100*_kept_seen/len(_seen):.0f}%")
print(f"Articles from the HELD-OUT domain ({HELD_OUT_DOMAIN}):  n={len(_unseen)}  "
      f"mean={sum(_unseen)/len(_unseen):.3f}  kept={100*_kept_unseen/len(_unseen):.0f}%")
print("\nI expected this to be the punchline -- 'quality classifiers are really")
print("domain classifiers' -- and it mostly is not. The unseen topic scores")
print("lower, but almost all of it still survives the cut. Topic transfer")
print("largely works. Whatever this classifier keys on, it is not subject")
print("matter. The next table is where it actually breaks.")

# %% [markdown]
# Now the part that matters. Every document below is high quality by any human
# standard. None of them looks like the positive class.

# %%
# A fair test has to control for document SHAPE. Every corpus document is eight
# sentences wrapped in navigation chrome; my first attempt fed the classifier
# bare three-sentence snippets and it rejected all of them, including good prose
# on a topic it knew. That was measuring length, not quality. So each probe below
# is padded to a comparable length and wrapped in identical chrome, leaving
# content type as the only variable.
_CHROME = NAV_CHROME[0]

CODE_SAMPLE = """def merge(left, right):
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    return out + left[i:] + right[j:]

def sort(xs):
    if len(xs) <= 1: return xs
    mid = len(xs) // 2
    return merge(sort(xs[:mid]), sort(xs[mid:]))

assert sort([3, 1, 2]) == [1, 2, 3]
assert sort([]) == []"""

LEGAL_SAMPLE = """Section 4.2. Indemnification. The Licensee shall indemnify, defend and hold
harmless the Licensor from and against any and all claims, damages, losses and
expenses, including reasonable attorneys' fees, arising out of or resulting from
the Licensee's breach of Section 3.1 hereof. Section 4.3. Limitation. In no
event shall either party be liable for indirect, incidental or consequential
damages, whether in contract or in tort, even if advised of the possibility
thereof. Section 4.4. Survival. The obligations set forth in this Article shall
survive the termination or expiration of this Agreement."""

MATH_SAMPLE = """Suppose that p is the largest prime number. Let q be defined as p factorial plus
one. No prime less than or equal to p divides q, since each such prime leaves a
remainder of one. Therefore q must have a prime factor strictly greater than p,
which contradicts the assumption that p was the largest. Hence there is no
largest prime, and the set of primes is infinite. The same argument shows that
the primes cannot be enumerated by any finite list, because the construction can
be repeated on any candidate list."""

CLINICAL_SAMPLE = """Patients were randomised in a one to one ratio to the intervention or to usual
care. The primary endpoint was mortality at twenty eight days after enrolment.
The adjusted odds ratio was 0.71 with a confidence interval of 0.54 to 0.94 and
a p value of 0.016. Adverse events occurred in 12.4 percent of the intervention
group and 11.9 percent of controls respectively. Loss to follow up was under
three percent in both arms. The trial was stopped early for efficacy after the
second interim analysis, in line with the prespecified stopping rule."""

OUT_OF_DOMAIN = [
    ("python source", CODE_SAMPLE),
    ("legal contract", LEGAL_SAMPLE),
    ("math proof", MATH_SAMPLE),
    ("clinical abstract", CLINICAL_SAMPLE),
    ("prose, seen domain", " ".join(DOMAINS["astronomy"][:8])),
    ("prose, held-out domain", " ".join(DOMAINS[HELD_OUT_DOMAIN][:8])),
]

print("\nSame classifier, out of domain")
print("-" * 74)
print("(identical chrome, comparable length -- content type is the only variable)")
print(f"{'document':<22} {'score':>9} {'corpus pctile':>15} {'verdict':>10}")
for name, text in OUT_OF_DOMAIN:
    wrapped = _wrap(text, _CHROME)          # identical chrome for every probe
    s = score(W, B, featurize(wrapped))
    verdict = "KEEP" if s >= QUALITY_THRESHOLD else "DISCARD"
    print(f"{name:<22} {s:>9.3f} {percentile_of(s):>14.0f}% {verdict:>10}")

print("\nRead that table against RAW_CRAWL_SAMPLES before drawing a conclusion.")
print("The two documents that get discarded -- source code and a contract -- are")
print("precisely the two genres represented in the NEGATIVE class. The proof and")
print("the clinical abstract survive, and neither was ever in the training data.")
print("\nSo the mechanism is not 'the classifier dislikes technical writing'. It")
print("is narrower and more actionable than that: the classifier deletes what")
print("the negative class taught it to delete, plus whatever shares that surface")
print("form. Prose survives on any topic, seen or unseen. Genres you swept into")
print("'raw crawl' do not.")
print("\nWhich means the negative class IS the corpus policy. Every decision")
print("about what counts as junk is encoded there, made once, usually by")
print("whoever assembled the training set, and written down nowhere. That is a")
print("policy document disguised as a dataset -- and it is why Chapter 3 keeps")
print("insisting you look at what a lab put in its negative class rather than")
print("at the accuracy it reports.")
print("\nAt corpus scale that quietly removes whole genres from the pre-training")
print("mix, and no metric you would normally look at moves. Labs that publish")
print("their recipes classify PER DOMAIN and mix the survivors rather than")
print("running one global threshold over everything.")

# %% [markdown]
# ## 7. Order of operations: dedup→filter or filter→dedup?
#
# Both pipelines contain the same two filters. Exercise 3.7 item 4 asks whether
# the final corpus differs. The operations look commutative. Let us measure
# instead of guessing.

# %%
def union_find_roots(doc_ids: list, pairs: set) -> dict:
    parent = {i: i for i in doc_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    return {i: find(i) for i in doc_ids}


perms128 = make_permutations(128)
sigs128 = {d: minhash(SHINGLE_SETS[d], perms128) for d in ids}
candidates = lsh_candidates(sigs128, 16, 8)
# Verify candidates exactly, as a production pipeline does.
verified = {(a, b) for a, b in candidates
            if jaccard(SHINGLE_SETS[a], SHINGLE_SETS[b]) >= TAU}

def dedup_keep_first(doc_ids: list) -> list:
    """Keep the lowest-numbered survivor of each connected component."""
    present = set(doc_ids)
    sub = {(a, b) for a, b in verified if a in present and b in present}
    roots = union_find_roots(doc_ids, sub)
    best: dict = {}
    for d in doc_ids:
        r = roots[d]
        if r not in best or d < best[r]:
            best[r] = d
    return sorted(best.values())


def filter_quality(doc_ids: list) -> list:
    return [d for d in doc_ids if quality[d] >= QUALITY_THRESHOLD]


deduped_first = dedup_keep_first(ids)
pipeline_a = filter_quality(deduped_first)        # dedup, then filter
pipeline_b = dedup_keep_first(filter_quality(ids))  # filter, then dedup
set_a, set_b = set(pipeline_a), set(pipeline_b)

print("\nTwo orderings of the same two filters")
print("-" * 74)
print(f"{'A: dedup -> quality filter':<34}: {len(pipeline_a)} documents")
print(f"{'B: quality filter -> dedup':<34}: {len(pipeline_b)} documents")
print(f"{'in A but not B':<34}: {len(set_a - set_b)}")
print(f"{'in B but not A':<34}: {len(set_b - set_a)}")
print(f"{'documents the classifier scored':<34}: "
      f"A={len(deduped_first)}  B={len(ids)}")

if set_a == set_b:
    print("\nOn this corpus the two orderings happened to agree. That is a")
    print("property of this sample, not a theorem -- see the argument below.")
else:
    lost = sorted(set_b - set_a)[:3]
    print("\nThe two corpora are NOT the same, and the mechanism is worth having.")
    print(f"Documents kept only by B, e.g. {lost}: their cluster's lowest-numbered")
    print("member was a low-quality mirror. Pipeline A elected that mirror as the")
    print("cluster representative and the quality filter then deleted it, taking")
    print("the whole cluster with it. Pipeline B removed the mirror first, so a")
    print("clean sibling got elected instead.")

print("\nDedup is a CHOICE OF REPRESENTATIVE, not just a removal, which is why")
print("the two do not commute. Production still deduplicates first, and the")
print("reason is the last row: the classifier runs on every surviving document,")
print("so dedup-first is the difference between scoring 10^10 and 10^9 documents.")
print("The standard fix is to make dedup quality-aware -- elect the")
print("HIGHEST-SCORING member of a cluster rather than the first one seen.")

# %% [markdown]
# ## 8. Things to try
#
# **1. Drop `SHINGLE_N` from 5 to 2 and rerun section 3.**
# *Common prediction:* near-duplicate detection gets more sensitive.
# *What happens:* it collapses. At $n=2$ every English document shares bigrams
# like "of the", so true Jaccard rises across unrelated pairs and the ground
# truth itself becomes garbage. You are not tuning sensitivity, you are
# redefining "same document".
#
# **2. Set `TAU` to 0.90 and rerun section 5.**
# *Common prediction:* precision improves.
# *What happens:* recall craters for every banding, because the knee is now far
# below the ground-truth cutoff and LSH is returning candidates you have just
# declared non-duplicates. `TAU` and $(b, r)$ must be chosen together — this is
# the most common LSH misconfiguration there is.
#
# **3. Remove `RAW_CRAWL_SAMPLES` from the negative class and rerun section 6.**
# *Common prediction:* out-of-domain scores shift a little.
# *What happens:* they all flip to KEEP and the finding vanishes. This is
# exactly the bug I shipped in the first draft. The lesson generalises: a
# quality classifier's behaviour is set by what you put in the negative class,
# and a negative class that is too easy produces a classifier that looks
# excellent and filters nothing.
#
# **4. Change `dedup_keep_first` to elect the highest-quality member.**
# *Common prediction:* pipeline A now matches pipeline B exactly.
# *What happens:* they converge but need not become identical, because quality
# filtering also changes which documents are *available* to be clustered.
# Working out the residual is worth ten minutes.
#
# **5. Raise `QUALITY_THRESHOLD` to 0.9 and rerun section 7.**
# *Common prediction:* a smaller corpus, same shape.
# *What happens:* the gap between the two orderings widens, because more
# elected representatives get deleted. Aggressive filtering makes pipeline
# order matter more, not less.

# %% [markdown]
# ## What to take away
#
# 1. **Exact-hash dedup catches only byte-identical documents.** In a corpus
#    where ~40% of documents live in a duplicate cluster it removes a small
#    fraction, because the web re-publishes content with a different cookie
#    banner attached.
# 2. **MinHash error falls as $1/\sqrt{k}$.** Slowly. Quadrupling $k$ halves
#    the error, which is why $k$ is 128 and why halving it costs more than
#    people expect.
# 3. **You do not choose a dedup threshold; you choose $b$ and $r$.** The knee
#    of $1 - (1 - J^r)^b$ sits near $(1/b)^{1/r}$ and everything follows from
#    that. Make sure the cutoff you *think* you set matches the banding.
# 4. **Bias LSH toward recall.** A false candidate costs one verification; a
#    missed duplicate stays in the corpus forever and is never observed.
# 5. **The negative class is the corpus policy.** A quality classifier
#    transfers across topic — unseen subjects in familiar prose survive — and
#    deletes the genres you swept into "raw crawl", plus anything sharing their
#    surface form. Which genres those are is a decision made once, by whoever
#    assembled the training set, and recorded nowhere. Read a lab's negative
#    class before you read its accuracy.
# 6. **Measure the thing you think you are measuring.** Three of this lab's
#    findings were artefacts — of an over-wide feature space, of an unrealistic
#    negative class, and of comparing documents of different lengths. Each one
#    produced a confident, plausible, wrong result first.
# 7. **Dedup and quality filtering do not commute**, because dedup elects a
#    representative. Production deduplicates first for cost, and repairs the
#    ordering bug by making the election quality-aware.
