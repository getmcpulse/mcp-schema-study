"""
Static analysis of MCP tool schemas.

Read `02_validate_ground_truth.py` first. Its result governs this file: the
registry preserves `properties`, per-parameter `description` and the tool
`description` exactly, and **drops the top-level `required` array**. So every
metric here is computed only on fields shown to survive, and nothing about
required-versus-optional is computed at all. It is not that the number would
be bad — it is that it would be the registry's number, not the servers'.

Definitions, all of them deliberately conservative
--------------------------------------------------
Each of these is a thing you can point at in the JSON, not an inference about
intent. Where a judgement call exists it is made in the direction that
*under*-reports the problem, so the figures are a floor.

  undescribed parameter
      a top-level entry of `inputSchema.properties` with no non-empty
      `description`. Nested object fields are not counted — a nested field is
      harder to blame on the author and would inflate the rate.

      This is **absence only**, and it is published as such. It is also a
      floor rather than the real rate: see the next entry.

  restates-the-name parameter          ← added after the post
      a parameter whose `description` is populated and, once the words of the
      parameter's own name and a short list of filler are removed, says
      nothing. `query` described as "The query", `maxResults` as "Maximum
      results". Arguably the commoner failure, and it passes every linter and
      the definition above.

      Counted separately rather than folded in, so the published figure stays
      the figure that was published. `substantively_undescribed` is the two
      together and is the number to read.

  undescribed tool
      no non-empty `description` on the tool itself.

  stated-but-unconstrained
      the parameter's own description *names its valid values* — "one of",
      "must be", a quoted list, a pipe-separated list — while the schema
      leaves it an unconstrained string with no `enum`. This is the strong
      version of the "should have been an enum" claim: we are not guessing
      that a set of values exists, the author wrote it down in prose and then
      did not put it in the schema.

  overlapping pair
      two tools in the same server whose descriptions exceed a cosine
      similarity threshold on tf-idf vectors. Reported as a curve across
      thresholds rather than at one magic number, because the number you pick
      is the finding otherwise.

  distinctive ratio                       ← the one to read
      of the content words in a tool's description, the share that appear in
      **no sibling tool's description on the same server**.

      This replaced cosine as the headline because cosine was answering the
      wrong question. Two tools on one server scored 0.98 and the pair looked
      damning until the text was read:

        3land_createCollection  "Create a new NFT collection on 3.Land
                                 marketplace. SAP MCP context: Protocol 3land;
                                 operation class write. Use for 3.Land NFT
                                 collection, minting, listing, cancellation…"
        3land_buyNFT            "Purchase an NFT from a 3.Land listing.
                                 SAP MCP context: Protocol 3land; operation
                                 class write. Use for 3.Land NFT collection,
                                 minting, listing, cancellation…"

      Those are not the same tool and the descriptions are not identical. What
      is true is narrower and more useful: the sentence that separates them is
      a small minority of what the model reads, and the rest is shared
      boilerplate the author added on purpose.

      The distinctive ratio states exactly that and needs no threshold to be
      meaningful. A ratio of 0 is the sharpest case in the corpus — nothing in
      the description distinguishes the tool from a sibling, and the tool name
      is the only signal a model has.

  input overlap                        ← added after the post
      for a *pair* of tools on one server, how much the set of things they
      operate on overlaps. Jaccard over each tool's input nouns.

      The ratio above asks whether two descriptions look alike. This asks
      whether two tools could both be *correct* for the same request, and that
      is the question that predicts a wrong selection.

      The evidence is a real ten-tool server for job applications. Three of
      its tools — `score_resume`, `analyze_job_description`, `optimize_resume`
      — score 50-60% distinctive, healthy by this corpus's own reckoning, and
      its author confirms they are ambiguous in practice: "check my resume for
      this job" is a fair sentence for all three. They are ambiguous because
      they all act on a resume and a job posting, not because they are worded
      alike. Distinctive share cannot see that and is not meant to.

      `looks_healthy_pairs` is the combination worth reporting: both
      descriptions above the corpus median and the pair still sharing its
      object. The existing metric has already called those healthy, so nothing
      else would ever mention them.

      An earlier attempt scored distinctiveness over the *output* half of each
      description instead. It failed against that same server — all ten verbs
      were lexically distinct, so it scored every tool at 100%, including the
      three the author had already flagged — and none of it survives.

      No part-of-speech tagger: input nouns are the description's content
      words minus the opening verb and minus anything after a return marker,
      plus every parameter name. Adjectives survive as "nouns", which dilutes
      an overlap rather than inventing one — a floor, like everything here.

  name distinctive ratio               ← added after the post
      the distinctive ratio again, over tool *names* rather than descriptions,
      with any server-wide prefix (`gmail_`, `aiapplyd_`) stripped first.

      Server authors report, independently, that when descriptions stop
      discriminating a model falls back to pattern-matching on the name. If
      that is what happens then the name is the backstop, and a server whose
      descriptions collide *and* whose names collide is in a different
      situation from one whose descriptions collide and whose names are clear.

      `quadrants` is the cross of the two. `both_low` — under 10% on the
      description and under 10% on the name — is the serious one: nothing left
      to discriminate on at all, and nobody had measured it.

Output
------
data/analysis.json   every figure, per pool
data/examples.json   real instances behind each finding, for spot-checking
"""

import json
import math
import os
import re
from collections import Counter, defaultdict

from stem import record, stem, surfaces

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "from", "at", "as", "is", "are", "be", "this", "that", "it", "its", "you",
    "your", "will", "can", "if", "when", "use", "used", "using", "returns",
    "return", "tool", "get", "set", "list", "all", "any", "not", "no", "only",
}

# The same list after stemming, because the list is written in surface forms
# and the counting happens in stem space. Without this, `returns` is filler and
# `returning` is a content word — the exact inconsistency the stemmer is here
# to remove. Nothing meaningful is lost: every word that lands on one of these
# stems is an inflection of a word already judged filler.
STOPWORD_STEMS = {stem(word) for word in STOPWORDS}

# ── "the description names the values, the schema does not constrain them" ──
#
# The first version of this matched any quoted list and was wrong about most
# of what it caught. On a 40-server sample it flagged 108 parameters, and the
# ones it flagged looked like this:
#
#     "Filter by line (e.g., \"1\", \"A\", \"F\")"
#     "Station ID or name (e.g., \"127\", \"Times Square\")"
#
# Those are illustrations, not closed sets — a station name is not an enum and
# never could be. Counting them would have produced a large, confident, wrong
# number, which is the specific failure this whole study is supposed to avoid.
#
# So the test is now two-sided. A description must use explicitly closed
# language, AND must not carry any marker that the list is illustrative. On
# the same sample that takes 108 down to 9, and all nine are real: `method`
# with "Options are: 'add' / 'remove'", `commitment` with
# "processed|confirmed|finalized", `use_case` with eight pipe-separated values.
#
# This is a floor, and the post says so. A parameter whose closed set is never
# written down anywhere cannot be detected from the schema at all.
EXAMPLE_MARKER = re.compile(
    r"\b(e\.?g\.?|for example|such as|examples?|like|including|etc\.?|"
    r"and more|among others)\b", re.I)

CLOSED_SET_PATTERNS = [
    re.compile(r"\b(?:one of|must be one of|must be either|either)\b\s*[:\-]?\s*['\"`\[(]", re.I),
    re.compile(r"\b(?:valid|allowed|possible|supported|accepted|permitted) values?\b", re.I),
    re.compile(r"\boptions?\s*:\s*['\"`\w]", re.I),
    # open | closed | pending
    re.compile(r"\b\w+\s*\|\s*\w+\s*\|\s*\w+\b"),
]


def content_stem(word):
    """A raw token -> its stem, or None when the token carries no content.

    Stopwords are tested on the *surface* form first and on the stem second.
    The first test is what the list was written for; the second is what makes
    it consistent once words are merged.
    """
    if len(word) <= 2 or word in STOPWORDS:
        return None
    key = record(word)
    return None if key in STOPWORD_STEMS else key


def tokenize(text):
    """Content words, stemmed. `screens` and `screening` are one word here.

    ── The bug this fixes ───────────────────────────────────────────────────
    This did exact string matching, so two tools sharing their most meaningful
    term — one writing `screens`, the other `screening` — each scored that term
    as distinctive. See `stem.py`. Stemming can only ever merge two words, so
    correcting it moves distinctiveness down and collision up, on every figure
    this function feeds.
    """
    keys = (content_stem(w)
            for w in re.findall(r"[a-z][a-z0-9_]+", (text or "").lower()))
    return [key for key in keys if key]


def cosine(a, b):
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    num = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def load():
    """Every collected row, deduplicated by server.

    ── Why this deduplicates ─────────────────────────────────────────────────
    The collector revisits servers while paginating the popular pool, so
    `tools.jsonl` contains the same `qualifiedName` more than once: 5,123 rows
    for 4,894 distinct servers. 104 of the 125 repeated names carry byte-
    identical tool lists — the same server collected twice — and the remaining
    21 pair a failed scrape (`tools == []`) with a successful one.

    Counting rows therefore inflated every total and, worse, weighted a server
    by how many times it happened to be collected. It landed almost entirely on
    one pool: the popular pool reported 465 servers and holds 263, a 77%
    overcount, while the long tail was clean at 4,486.

    The rates barely moved when this was fixed — undescribed parameters 21.8% ->
    21.5%, zero-distinctive tools 17.4% -> 17.7% — which is why it went unnoticed.
    The counts moved a lot: 4,951 servers -> 4,749, 87,146 tools -> 82,549.

    ── Why it keeps the row with tools ───────────────────────────────────────
    For the 21 names where the two rows differ, one is a scrape that returned
    nothing. Keeping the longer list keeps the server in the corpus rather than
    discarding it as tool-less.

    That is not, however, what moved `tools == []` from 13 to 8. Checked: none
    of those 8 names published tools in any other row. The 13 was 8 servers
    counted repeatedly — one of them four times — so the drop is the same
    double-count as everywhere else, not a rescue.
    """
    best = {}
    order = []

    with open(os.path.join(DATA, "tools.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            name = row.get("qualifiedName")
            existing = best.get(name)

            if existing is None:
                best[name] = row
                order.append(name)
                continue

            # Prefer whichever row actually came back with tools.
            if len(row.get("tools") or []) > len(existing.get("tools") or []):
                best[name] = row

    return [best[name] for name in order]


def describes_its_values(text):
    """True only when the description states a closed set and gives no sign
    that the set is merely illustrative."""
    text = text or ""
    if not any(p.search(text) for p in CLOSED_SET_PATTERNS):
        return False
    return not EXAMPLE_MARKER.search(text)


# ── what a tool operates on ─────────────────────────────────────────────────
#
# The words that announce a return, used only to *exclude* the returned half
# from the input set. Nothing is scored on the output: that was tried, and it
# failed — see the `input_overlap` entry in the module docstring.
RETURN_WORDS = set("""
    return returns returning returned give gives giving given provide provides
    providing provided produce produces producing produced output outputs
    outputting yield yields yielding respond responds responding
""".split())

# Same reason as `STOPWORD_STEMS`: the list is written out in every inflection,
# and once words are merged the test has to happen on the merged form or
# `provides` and `providing` stop being the same exclusion.
RETURN_STEMS = {stem(word) for word in RETURN_WORDS}

RETURN_MARKER = re.compile(
    r"\b(?:returns?|returning|returned|gives?|giving|provides?|providing|"
    r"produces?|producing|outputs?|yields?|responds?\s+with)\b|:", re.I)

# Above this, two tools act on enough of the same thing to be confusable. Half
# the nouns either of them uses: lower and one shared object — every tool on a
# mail server mentions a message — flags the whole server; higher and a cluster
# that shares its object and differs on everything else stops being visible.
INPUT_OVERLAP_FLAG = 0.5

# The study's own "low" mark, applied to descriptions and to names alike, so
# "both low" means the same thing on both halves.
LOW_DISTINCTIVE = 0.1

# A first token has to be on this share of a server's tools to be a namespace
# rather than a collision.
PREFIX_SHARE = 0.8

# The corpus median distinctive share, used as the bar for `looks_healthy_pairs`:
# a pair above it has already been called healthy by the headline metric, so a
# flag there is the one nothing else in the study would ever mention.
#
# Written out rather than computed in the same pass, so the bar does not move
# under a subset — the six size buckets each have their own median and
# comparing a bucket against itself would say nothing. It was 0.26; stemming
# took it to 0.20, and this line is the one that has to be edited by hand when a
# re-run moves it again. `distinctive_ratio.median` on the `all` pool is the
# value it must equal.
CORPUS_MEDIAN_DISTINCTIVE = 0.20


def all_words(text):
    """Tokens without the stopword filter, in order.

    `tokenize` drops `get`, `set`, `list` and `return` as filler, which is right
    for a content-word ratio and wrong for finding the opening verb — those are
    among the commonest verbs on the registry.
    """
    return re.findall(r"[a-z][a-z0-9_]+", (text or "").lower())


def input_nouns(tool):
    """The words a tool operates on.

    Content words from the description, minus the returned half and minus the
    opening verb, plus every parameter name. See the module docstring for why
    each of those two exclusions is there and which way it errs.

    The parameter names do most of the work: a parameter name is the one place
    an author states the object with no prose around it, and it needs no
    tagging at all.
    """
    desc = tool.get("description") or ""

    before = desc
    marker = RETURN_MARKER.search(desc)
    # Only when something comes before it. "Return the user's job matches" opens
    # on its verb, not on a return marker, and treating that as output would
    # leave the tool with no nouns at all.
    if marker and re.search(r"[a-z]", desc[:marker.start()], re.I):
        rest = desc[marker.start():]
        stop = re.search(r"[.!?\n]", rest)
        before = desc[:marker.start()] + (rest[stop.start():] if stop else "")

    # The verb is found by position, so the ordering pass runs on raw tokens and
    # the merge happens after it. Stemming first would not change which token is
    # first, but it would make the RETURN_WORDS test read against a form the
    # list was not written in.
    ordered = [w for w in all_words(before) if content_stem(w)]
    nouns = {content_stem(w) for w in ordered[1:]
             if content_stem(w) not in RETURN_STEMS}

    props = (tool.get("inputSchema") or {}).get("properties") or {}
    for param in props:
        # Already stemmed — `name_tokens` runs `normalise`, which expands the
        # abbreviation and then stems. So `resumes` in a sentence and
        # `resume_text` in a schema land on one noun, which is the whole point.
        nouns.update(t for t in name_tokens(param) if len(t) > 1)

    return nouns


def jaccard(a, b):
    """Shared nouns over the nouns either tool uses."""
    if not a or not b:
        return 0.0
    shared = len(a & b)
    return shared / (len(a) + len(b) - shared)


# ── U: synonym verbs, which exact matching misses even after stemming ───────
#
# `get_order` and `fetch_order` are the same tool name written twice. No stemmer
# reaches that — `get` and `fetch` share no letters — so a synonym table is the
# only thing that can, and the table is small because the failure is
# concentrated: `get` leads 15.6% of all public tool names, and the four sets
# below cover nearly every collision of this kind in the corpus.
#
# ── Applied to the leading verb only, and why ──────────────────────────────
# A name is a verb followed by an object. Collapsing synonyms anywhere in the
# name would merge `get_search_index` with `search_get_index`, which are not the
# same tool and do not read as the same tool; collapsing only the first token
# merges exactly the pairs a model is being asked to choose between.
#
# ── Deliberately short ─────────────────────────────────────────────────────
# Every verb left off this table counts as distinct, so the table under-reports
# collisions rather than inventing them — the direction everything in this file
# errs in. The four sets are the ones multiple server authors named independently.
LEADING_VERB_SYNONYMS = {
    # read one thing
    "get": "get", "fetch": "get", "read": "get", "retrieve": "get", "load": "get",
    # read many
    "list": "list", "index": "list", "enumerate": "list", "browse": "list",
    # read many, by predicate
    "search": "search", "query": "search", "find": "search", "lookup": "search",
    # unmake
    "delete": "delete", "remove": "delete", "destroy": "delete", "purge": "delete",
    # make
    "create": "create", "add": "create", "insert": "create", "new": "create",
    # change
    "update": "update", "modify": "update", "edit": "update", "patch": "update",
}

#: Keyed on stems, because that is the space names are counted in. Built once.
VERB_CANONICAL = {record(word): record(canonical)
                  for word, canonical in LEADING_VERB_SYNONYMS.items()}

#: `getAll` tokenises to `get` + `all`, so the leading-token rule alone would
#: canonicalise it to `get` and leave it distinct from `list`. It is the one
#: two-token verb common enough to be worth the special case.
GET_ALL = (record("get"), record("all"))


def canonical_verb(words):
    """A name's tokens with its leading verb canonicalised.

    Returns the list unchanged when the first token is not a verb this table
    knows, which is the common case and the safe one.
    """
    if not words:
        return words

    if len(words) >= 2 and (words[0], words[1]) == GET_ALL:
        return [record("list")] + list(words[2:])

    canonical = VERB_CANONICAL.get(words[0])
    return [canonical] + list(words[1:]) if canonical else list(words)


def name_words(name):
    """`Gmail_DeleteDraftEmail` -> `['gmail', 'delete', 'draft', 'email']`."""
    return [t for t in name_tokens(name or "") if len(t) > 1]


def common_prefix(names):
    """The token nearly every tool on the server starts with, if there is one.

    Only the first token, and only at `PREFIX_SHARE`. A word that appears in
    most names further along — `message` on a mail server — is a real collision
    rather than a namespace, and stripping it would hide the finding.
    """
    if len(names) < 3:
        return None

    first = [w[0] for w in (name_words(n) for n in names) if w]
    if len(first) < len(names) * PREFIX_SHARE:
        return None

    token, count = Counter(first).most_common(1)[0]
    return token if count >= len(names) * PREFIX_SHARE else None


# ── "the description is populated and says nothing" ─────────────────────────
#
# Filler, removed before a parameter description is called substantive. Two
# kinds, both here for one reason. The first is ordinary English glue. The
# second is the words that name the *slot* rather than the value in it —
# `value`, `field`, `apply`, `specify`. "A filter to apply" on a parameter
# named `filter` is three filler words around the parameter's own name.
#
# The list is a judgement call and deliberately short: every word left off it
# counts as information, so a description survives as described on the strength
# of one real word. That under-reports the problem, which is the direction this
# file errs in everywhere else.
PARAM_FILLER = set("""
    a an the of to for this that is are be and or in on at by with from it its as
    use used using uses optional required
    apply applies applied applying specify specifies specified specifying
    provide provides provided pass passed passing give gives given
    set sets setting want wants desired need needs
    value values field fields parameter parameters param params
    argument arguments input inputs option options here there which what your you
""".split())

# Short forms a name and its description spell differently. `maxResults`
# described as "Maximum results" is the same two words twice, and a token
# comparison misses it because `max` is not `maximum`. One map catches it
# without letting a prefix rule loose — a prefix rule would also collapse
# `user` into `username`, which is a real distinction a description may draw.
ABBREVIATIONS = {
    "max": "maximum", "min": "minimum", "num": "number", "cnt": "count",
    "qty": "quantity", "amt": "amount", "pct": "percent",
    "id": "identifier", "uid": "identifier", "idx": "index", "len": "length",
    "desc": "description", "msg": "message", "addr": "address",
    "ts": "timestamp", "tz": "timezone",
    "cfg": "configuration", "config": "configuration", "dir": "directory",
    "req": "request", "res": "response", "resp": "response",
    "str": "string", "int": "integer", "bool": "boolean", "args": "argument",
    "auth": "authentication", "repo": "repository",
    "org": "organisation", "organization": "organisation",
    "prev": "previous", "curr": "current", "info": "information",
    "spec": "specification", "attr": "attribute", "prop": "property",
    "ref": "reference", "src": "source", "dst": "destination",
    "dest": "destination",
}


def normalise(word):
    """`max`, `Results` -> `maximum`, `result`. Expand, then stem.

    The order matters and it is not interchangeable. `organization` and
    `organisation` stem to `organ` and `organis` — two spellings of one word,
    pulled further apart by the stemmer rather than together — so the map has
    to collapse them *before* Porter sees either. The map is also the only
    thing that knows `max` and `maximum` are the same word; no suffix rule can
    reach that, and a prefix rule loose enough to try would also collapse
    `user` into `username`, which is a distinction a description may draw.

    Three hand-rolled plural rules used to live here. They were a stemmer with
    one rule in it, and they are gone: `screens` and `screening` were the pair
    they could not merge.
    """
    return stem(ABBREVIATIONS.get(word, word))


def name_tokens(name):
    """`maxResults` -> `['maximum', 'result']`. camelCase, snake_case, kebab."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")
    # `HTTPServer` -> `HTTP Server`, so an acronym does not swallow the word
    # after it.
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return [normalise(t.lower()) for t in re.split(r"[^A-Za-z0-9]+", spaced) if t]


def param_state(name, desc):
    """`missing`, `restates_name` or `described` — see the module docstring."""
    if not (desc or "").strip():
        return "missing"

    from_name = set(name_tokens(name))
    remaining = [
        normalise(w) for w in re.split(r"[^a-z0-9]+", desc.lower())
        if w and w not in PARAM_FILLER
    ]
    return "described" if [w for w in remaining if w not in from_name] else "restates_name"


# ── C: what a tool says it gives back ───────────────────────────────────────
#
# Input overlap on its own produces false positives on well-designed pairs, and
# the counter-example is concrete. A six-tool ticket server has two tools with
# identical input shapes that are correctly two tools: one returns flat details
# for many tickets, the other a bounded evidence bundle with comments and source
# anchors. Same input, clearly different stated output. Input overlap flags it
# and the author is right to dismiss the flag.
#
# So the rule is a conjunction, not a single signal:
#
#     overlapping inputs + distinguishable outputs   = fine design
#     overlapping inputs + indistinguishable outputs = the failure
#
# On a large server a 10% false-positive rate buries the signal entirely, which
# is the whole reason this gate exists rather than a warning in the docs.

#: Below this, two tools' stated outputs are different enough to tell apart.
#: Same Jaccard, same scale as the input threshold, so "half the terms either
#: one uses" means the same thing on both halves of the rule.
OUTPUT_DISTINCT_FLAG = 0.5


def output_terms(tool):
    """The words a tool says it gives back.

    Three sources, in the order they are worth trusting:

      1. `outputSchema` property names, when there is one. Like a parameter
         name on the input side, this is the author stating the shape with no
         prose around it.
      2. Everything after a return marker — `returns`, `produces`, `gives`,
         `provides`, or a colon — up to the end of that sentence. This is the
         exact half `input_nouns` throws away, read here instead of discarded.
      3. Nothing. Which is a finding rather than a gap: see below.

    An empty set means the tool never says what it returns, and that is not the
    same as "could not be parsed". A tool that does not state its output cannot
    be told apart from a sibling by its output, so silence is treated as
    indistinguishable rather than as missing data. It is also the commonest
    shape on a focused server, which is the gap `D` exists to report.
    """
    terms = set()

    schema = tool.get("outputSchema") or {}
    for name in (schema.get("properties") or {}):
        terms.update(t for t in name_tokens(name) if len(t) > 1)

    desc = tool.get("description") or ""
    marker = RETURN_MARKER.search(desc)
    # Only when something comes before it, matching `input_nouns`: "Return the
    # user's job matches" opens on its verb, and the whole description is the
    # output rather than the half after a marker that is not there.
    if marker and re.search(r"[a-z]", desc[:marker.start()], re.I):
        rest = desc[marker.start():]
        stop = re.search(r"[.!?\n]", rest)
        tail = rest[:stop.start()] if stop else rest
        terms.update(k for k in (content_stem(w) for w in all_words(tail))
                     if k and k not in RETURN_STEMS)

    return terms


def outputs_distinguishable(a, b):
    """True when two tools state outputs a reader could tell apart.

    Silence on either side is False, not True. The failure this whole gate is
    protecting is "two tools that act on the same object and never say what
    they return"; scoring an unstated output as distinguishable would wave
    exactly that case through.
    """
    if not a or not b:
        return False
    return jaccard(a, b) < OUTPUT_DISTINCT_FLAG


# ── D: the domain noun and the cluster noun are different things ────────────
#
# One reader pointed out that distinctive share penalises focused servers: a
# memory server says "memory" in every description by design, and that is the
# server's name repeated rather than a collision. Correct.
#
# But stripping it globally breaks input overlap, because on a focused server
# the collision *is* the domain noun. Strip `resume` from the job-applications
# server and you delete exactly what makes those three tools confusable.
#
# The resolution removes the threshold question rather than answering it.
# Frequency is not a dial; it is a distinction the count already makes:
#
#     a noun in ALL of a server's tools   = the domain. The server's name,
#                                           repeated. Not a collision.
#     a noun in a SUBSET of the tools     = a cluster, and that subset is the
#                                           candidate collision list.
#
# So there are two readings with different preprocessing, and they are not
# averaged. THE GAP BETWEEN THEM IS THE FINDING: clean on reading 1 and bad on
# reading 2 is a focused server whose tools all act on one object and never say
# what they return. Common shape, and averaging the two hides it completely.


# ── F: the published rate counts the top level only ────────────────────────
#
# `undescribed parameter` is defined as a top-level entry of
# `inputSchema.properties` with no description, and 21.5% is that number. The
# definition was chosen deliberately — a nested field is harder to lay at the
# author's door — but it makes the figure a floor for a second reason, on top of
# the restatements: a tool whose top-level parameters are all described and whose
# nested object is entirely bare scores clean.
#
# An undescribed field at depth four is exactly as invisible to a model as one
# at depth one. So both are computed, and they are kept apart: the top-level
# figure stays the headline, because it is the one that is comparable with what
# was published, and the all-depths figure is reported beside it.

#: How far down to walk. Deeper than any real MCP schema, and a bound rather
#: than a belief: a `$ref` resolver can hand back a structure that recurses.
MAX_SCHEMA_DEPTH = 8


def deep_params(tool):
    """Every parameter at every depth, as `(path, name, schema)`.

    Walks `properties`, `items`, and the `anyOf`/`oneOf`/`allOf` branches,
    because a parameter defined through a union is still a parameter.

    Paths are dotted and array steps are marked: `filter.from`, `tags[].label`,
    `options.retry.maxAttempts`. That is the whole reason to do this — a count
    is only actionable if it can say which field.
    """
    out = []
    seen = set()

    def walk(node, path, depth):
        if not isinstance(node, dict) or depth > MAX_SCHEMA_DEPTH:
            return
        # A generator that resolves `$ref` can hand back the same object twice,
        # and visiting it twice would double every parameter beneath it.
        if id(node) in seen:
            return
        seen.add(id(node))

        props = node.get("properties")
        if isinstance(props, dict):
            for key, child in props.items():
                if not isinstance(child, dict):
                    child = {}
                child_path = f"{path}.{key}" if path else key
                out.append((child_path, key, child))
                walk(child, child_path, depth + 1)

        items = node.get("items")
        if items is not None:
            walk(items, f"{path}[]" if path else "[]", depth + 1)

        for keyword in ("anyOf", "oneOf", "allOf"):
            branch = node.get(keyword)
            if isinstance(branch, list):
                for sub in branch:
                    walk(sub, path, depth + 1)

    walk(tool.get("inputSchema") or {}, "", 0)
    return out


# ── H: depth is a max, and a max conflates two different costs ─────────────
#
# A tool with one field at depth 5 and a tool with forty fields at depth 5 score
# identically on depth, and they are not the same job. Two things are worth
# separating:
#
#   **how much there is to fill in** — the leaf count. A deterministic
#   five-level object is tedious and a model will get it right.
#
#   **how much of it is a choice** — the branch points. A depth-two schema with
#   two overlapping `oneOf` branches is a decision the model can get wrong, and
#   depth cannot see it at all.
#
# ── What cannot be computed here, and why it is not attempted ──────────────
# Conditional requirements — `if this field is X, that one becomes required` —
# are the sharpest version of branch ambiguity and are unreachable from this
# corpus. The registry strips the top-level `required` array; that is established
# in `02_validate_ground_truth.py` by booting the four reference servers, 27 of
# whose 37 tools emit one against the registry's none. So branch scoring below
# uses property sets only, and says so, rather than quietly reporting a weaker
# number under a stronger name.

#: Branches that overlap on more than this share of their properties are not a
#: decision a reader could make confidently.
BRANCH_AMBIGUOUS = 0.5


def schema_shape(tool):
    """Leaf count, structural depth and decision depth for one tool.

    `structural_depth` is the existing max-over-the-tree. `decision_depth` is
    how many branch points sit on the path that reaches deepest — the part of
    the depth that is a choice rather than a walk.
    """
    leaves = 0
    max_depth = 0
    max_decisions = 0
    seen = set()

    def walk(node, depth, decisions):
        nonlocal leaves, max_depth, max_decisions
        if not isinstance(node, dict) or depth > MAX_SCHEMA_DEPTH:
            return
        if id(node) in seen:
            return
        seen.add(id(node))

        props = node.get("properties")
        items = node.get("items")
        branches = [b for k in ("anyOf", "oneOf") for b in (node.get(k) or [])
                    if isinstance(b, dict)]

        child_decisions = decisions + (1 if len(branches) >= 2 else 0)
        has_child = False

        if isinstance(props, dict) and props:
            has_child = True
            for child in props.values():
                walk(child if isinstance(child, dict) else {}, depth + 1, child_decisions)
        if isinstance(items, dict):
            has_child = True
            walk(items, depth + 1, child_decisions)
        for branch in branches:
            has_child = True
            walk(branch, depth + 1, child_decisions)
        for branch in (node.get("allOf") or []):
            if isinstance(branch, dict):
                has_child = True
                walk(branch, depth + 1, child_decisions)

        if not has_child:
            # A leaf: something the author has to put a value in.
            leaves += 1
            max_depth = max(max_depth, depth)
            max_decisions = max(max_decisions, child_decisions)

    walk(tool.get("inputSchema") or {}, 0, 0)
    return {"leaves": leaves, "depth": max_depth, "decisions": max_decisions}


def branch_points(tool):
    """Every `anyOf`/`oneOf` in the schema, scored for how hard the choice is.

    Branch *count* is the wrong measure on its own: `string or null` is two
    branches and no decision at all, while three structurally similar objects is
    two branches and a real one. So a branch point is scored on what a model
    would actually have to go on.

      * **A discriminator makes it free.** A property present in every branch
        pinned to a single value — `type: "email"` against `type: "sms"` — is
        exactly how a reader tells them apart, and the choice is not ambiguous.
      * **Scalars are not a decision.** Branches with no properties at all are a
        nullable or a union of primitives.
      * **Otherwise, how much the property sets overlap**, averaged over every
        pair of branches. Two objects sharing most of their fields are two
        objects a model will pick between at random.

    Required-set overlap would be the strongest term here and is not available:
    the registry strips `required`. Stated in the module docs and not estimated.
    """
    out = []
    seen = set()

    def properties_of(branch):
        props = branch.get("properties")
        return set(props) if isinstance(props, dict) else set()

    def discriminated(branches):
        """A property in every branch, pinned to one value in each."""
        common = None
        for branch in branches:
            pinned = set()
            props = branch.get("properties")
            if isinstance(props, dict):
                for name, child in props.items():
                    if not isinstance(child, dict):
                        continue
                    if "const" in child or (isinstance(child.get("enum"), list)
                                            and len(child["enum"]) == 1):
                        pinned.add(name)
            common = pinned if common is None else (common & pinned)
            if not common:
                return False
        return bool(common)

    def walk(node, path, depth):
        if not isinstance(node, dict) or depth > MAX_SCHEMA_DEPTH or id(node) in seen:
            return
        seen.add(id(node))

        for keyword in ("anyOf", "oneOf"):
            branches = [b for b in (node.get(keyword) or []) if isinstance(b, dict)]
            if len(branches) < 2:
                continue

            sets = [properties_of(b) for b in branches]
            if not any(sets):
                score = 0.0            # a nullable, or a union of primitives
            elif discriminated(branches):
                score = 0.0            # the author said which is which
            else:
                overlaps = [jaccard(sets[i], sets[j])
                            for i in range(len(sets)) for j in range(i + 1, len(sets))]
                score = sum(overlaps) / len(overlaps) if overlaps else 0.0

            out.append({"path": path or "(root)", "keyword": keyword,
                        "branches": len(branches), "ambiguity": round(score, 3)})

        props = node.get("properties")
        if isinstance(props, dict):
            for name, child in props.items():
                walk(child if isinstance(child, dict) else {},
                     f"{path}.{name}" if path else name, depth + 1)
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, f"{path}[]" if path else "[]", depth + 1)
        for keyword in ("anyOf", "oneOf", "allOf"):
            for branch in (node.get(keyword) or []):
                if isinstance(branch, dict):
                    walk(branch, path, depth + 1)

    walk(tool.get("inputSchema") or {}, "", 0)
    return out


def document_frequency(token_sets):
    """How many tools on this server use each word.

    Every distinctiveness figure here is "the words no sibling also uses", and
    the obvious way to compute that is to union every *other* tool's words and
    subtract. That is O(n2) with a set union inside it, and it degrades exactly
    where the study cares most: the largest server in the corpus publishes 2,530
    tools, which is 2,530 unions of 2,529 sets each.

    One tally answers the same question in a single pass. A word is distinctive
    to its tool when exactly one tool uses it, shared when more than one does,
    and the server's domain when every one does — three readings of one count.
    """
    df = Counter()
    for words in token_sets:
        df.update(words)
    return df


def domain_nouns(token_sets):
    """The words every single tool on this server uses.

    Every tool, not nearly every tool. The point of the rule is that it has no
    dial in it: a ratio would put the finding back inside a threshold nobody can
    defend, and the count already separates the two cases cleanly.

    The cost is that a noun on 59 of 60 tools reads as a cluster rather than the
    domain. That is the right way to be wrong — it is reported *as* a cluster of
    59, where its size is visible on the page, rather than silently deleted.
    """
    if len(token_sets) < 2:
        return set()
    shared = set.intersection(*token_sets) if all(token_sets) else set()
    return shared


def clusters(names, token_sets, domain):
    """Every noun on a subset of the tools, and which tools share it.

    The candidate collision list, emitted directly rather than rolled into a
    score. On the job-applications server the tightest cluster is `ats`, shared
    by `score_resume` and `optimize_resume` and nothing else — which is the pair
    its author flagged first, and the pair no ratio in this file names.

    Sorted tightest first: a noun on two tools of ten localises the problem, a
    noun on nine of ten barely narrows it.
    """
    members = defaultdict(list)
    for name, words in zip(names, token_sets):
        for word in words:
            if word not in domain:
                members[word].append(name)

    out = [(word, tools) for word, tools in members.items() if len(tools) >= 2]
    out.sort(key=lambda item: (len(item[1]), item[0]))
    return out


def median(xs):
    """The middle of a sorted list. Module level because the per-server loop
    needs it too, and a closure defined after that loop is not yet bound when
    the loop runs."""
    return xs[len(xs) // 2] if xs else None


def pctile(xs, q):
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None


def analyse_pool(servers, idf, examples):
    """Every figure for one pool of servers."""
    n_servers = len(servers)
    tools_per_server = []
    params_per_tool = []
    # What the tool list costs before anyone asks a question. Every byte here
    # is sent on every connection, so it is the one cost that is paid whether
    # a tool is used or not — and unlike everything else in this file it needs
    # no interpretation at all: it is `len(json.dumps(...))`.
    schema_bytes_per_server = []

    n_tools = 0
    undescribed_tools = 0
    n_params = 0
    undescribed_params = 0
    # Populated, and says nothing the parameter's own name did not. Counted
    # apart from the line above so the published absence-only rate stays the
    # rate that was published.
    restated_params = 0
    servers_with_restated_param = 0

    # ── F: the same three counts, at every depth ─────────────────────────────
    # Reported beside the top-level figures, never merged into them. The
    # top-level denominator is what the published rate is comparable against;
    # this one is what an author actually has to go and fix.
    # ── H: the two costs depth conflates ────────────────────────────────────
    leaf_counts = []
    structural_depths = []
    decision_depths = []
    tools_with_branches = 0
    branch_points_total = 0
    ambiguous_branches = 0
    servers_with_ambiguous_branch = 0

    deep_params_total = 0
    deep_undescribed = 0
    deep_restated = 0
    # Tools whose top level is spotless and whose nested fields are not — the
    # specific case the published definition scores as clean.
    tools_clean_on_top_bare_below = 0
    servers_with_nested_gap = 0

    string_params = 0
    enum_params = 0
    stated_unconstrained = 0

    servers_with_undescribed_param = 0
    servers_with_undescribed_tool = 0
    servers_with_stated_unconstrained = 0
    servers_with_dup_names = 0

    # overlap measured across a range, so the threshold is not the finding
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    servers_over = {t: 0 for t in thresholds}
    pairs_over = {t: 0 for t in thresholds}
    total_pairs = 0

    # distinctive ratio — see the module docstring
    distinctive = []
    servers_with_zero_distinctive = 0

    # input overlap — whether two tools could both be correct for one request,
    # gated on whether they state outputs anyone could tell apart
    overlap_pairs = 0
    overlap_pairs_sharing_inputs = 0
    overlap_pairs_flagged = 0
    overlap_pairs_near_miss = 0
    overlap_pairs_looks_healthy = 0
    servers_with_overlap = 0
    servers_with_near_miss = 0
    servers_with_looks_healthy = 0

    # ── D: the domain noun and the cluster noun ──────────────────────────────
    # Two readings, kept apart on purpose. Reading 1 strips the words every tool
    # on the server uses and asks whether the descriptions look alike; reading 2
    # keeps them and asks whether two tools could be right for one sentence. The
    # gap between them is the finding, and averaging them hides it.
    domain_stripped = []
    servers_with_domain = 0
    domain_noun_count = 0
    # Clean on reading 1, bad on reading 2: a focused server whose tools all act
    # on one object and never say what they return.
    servers_in_the_gap = 0

    # name distinctiveness, and the cross of it with the description figure
    name_distinctive = []
    servers_with_prefix = 0
    quadrant = Counter()

    for s in servers:
        tools = s.get("tools") or []
        tools_per_server.append(len(tools))
        n_tools += len(tools)
        # `outputSchema` is excluded: it is not always sent to the model, and
        # counting it would overstate the bill on the servers that publish one.
        schema_bytes_per_server.append(len(json.dumps(
            [{k: v for k, v in t.items() if k != "outputSchema"} for t in tools],
            separators=(",", ":"))))

        names = [t.get("name") for t in tools]
        if len(names) != len(set(names)):
            servers_with_dup_names += 1

        s_undesc_param = s_undesc_tool = s_stated = False
        s_restated_param = False
        s_nested_gap = False
        s_ambiguous_branch = False

        vectors = []
        for t in tools:
            desc = (t.get("description") or "").strip()
            if not desc:
                undescribed_tools += 1
                s_undesc_tool = True
                if len(examples["undescribed_tool"]) < 25:
                    examples["undescribed_tool"].append(
                        {"server": s["qualifiedName"], "tool": t.get("name")})

            tf = Counter(tokenize(desc))
            vectors.append({w: (1 + math.log(c)) * idf.get(w, 0.0) for w, c in tf.items()})

            props = (t.get("inputSchema") or {}).get("properties") or {}
            params_per_tool.append(len(props))

            for pname, p in props.items():
                if not isinstance(p, dict):
                    continue
                n_params += 1
                pdesc = (p.get("description") or "").strip()
                state = param_state(pname, pdesc)
                if state == "missing":
                    undescribed_params += 1
                    s_undesc_param = True
                    if len(examples["undescribed_param"]) < 25:
                        examples["undescribed_param"].append(
                            {"server": s["qualifiedName"], "tool": t.get("name"),
                             "param": pname})
                elif state == "restates_name":
                    restated_params += 1
                    s_restated_param = True
                    if len(examples["restates_name"]) < 25:
                        examples["restates_name"].append(
                            {"server": s["qualifiedName"], "tool": t.get("name"),
                             "param": pname, "description": pdesc[:220]})

                ptype = p.get("type")
                is_string = ptype == "string" or (
                    isinstance(ptype, list) and "string" in ptype)
                has_enum = bool(p.get("enum"))
                if is_string:
                    string_params += 1
                    if has_enum:
                        enum_params += 1
                    elif describes_its_values(pdesc):
                        stated_unconstrained += 1
                        s_stated = True
                        if len(examples["stated_unconstrained"]) < 25:
                            examples["stated_unconstrained"].append(
                                {"server": s["qualifiedName"], "tool": t.get("name"),
                                 "param": pname, "description": pdesc[:220]})
                elif has_enum:
                    enum_params += 1

            # ── H: size, depth, and how much of the depth is a choice ───────
            shape = schema_shape(t)
            leaf_counts.append(shape["leaves"])
            structural_depths.append(shape["depth"])
            decision_depths.append(shape["decisions"])

            points = branch_points(t)
            if points:
                tools_with_branches += 1
                branch_points_total += len(points)
                worst = max(points, key=lambda b: b["ambiguity"])
                if worst["ambiguity"] > BRANCH_AMBIGUOUS:
                    ambiguous_branches += 1
                    s_ambiguous_branch = True
                    if len(examples["branch_ambiguity"]) < 25:
                        examples["branch_ambiguity"].append({
                            "server": s["qualifiedName"],
                            "tool": t.get("name"),
                            "path": worst["path"],
                            "keyword": worst["keyword"],
                            "branches": worst["branches"],
                            "ambiguity": worst["ambiguity"],
                        })

            # ── F: walk the whole schema, not just its first level ──────────
            top_level_paths = set(props)
            nested_missing = []
            for path, key, child in deep_params(t):
                deep_params_total += 1
                state = param_state(key, (child.get("description") or "").strip())
                if state == "missing":
                    deep_undescribed += 1
                elif state == "restates_name":
                    deep_restated += 1
                # Nested only: the top level is already counted above, and this
                # is the population the headline figure cannot see.
                if state != "described" and path not in top_level_paths:
                    nested_missing.append(path)

            if nested_missing and not any(
                param_state(k, (v.get("description") or "").strip()) != "described"
                for k, v in props.items() if isinstance(v, dict)
            ):
                # Every top-level parameter is described and something below is
                # not. This tool scores perfect on the published definition.
                tools_clean_on_top_bare_below += 1
                s_nested_gap = True
                if len(examples["nested_only"]) < 25:
                    examples["nested_only"].append({
                        "server": s["qualifiedName"],
                        "tool": t.get("name"),
                        "top_level_params": len(props),
                        "undescribed_below": nested_missing[:8],
                    })

        # ── distinctive ratio, per tool, against its own siblings ──────────
        # Only meaningful with something to be distinct *from*, so
        # single-tool servers are skipped rather than scored 1.0.
        if len(tools) >= 2:
            token_sets = [set(tokenize(t.get("description") or "")) for t in tools]

            # ── D reading 1: distinctiveness with the domain taken out ──────
            # A memory server says "memory" in every description by design. That
            # is the server's name repeated, not a collision, and counting it
            # against every tool penalises the server for being focused.
            domain = domain_nouns([s for s in token_sets if s])
            if domain:
                servers_with_domain += 1
                domain_noun_count += len(domain)
                if len(examples["domain_nouns"]) < 25:
                    examples["domain_nouns"].append({
                        "server": s["qualifiedName"],
                        "tools": len(tools),
                        "domain": sorted(surfaces(domain))[:12],
                    })

            # The same tally as the un-stripped reading, over stripped sets.
            # Domain words come off both sides at once, so a word shared only
            # through the domain is not counted against the tool twice.
            stripped_df = document_frequency([w - domain for w in token_sets])
            for words in token_sets:
                if not words:
                    continue
                stripped = words - domain
                # Every content word is the domain. The tool has said nothing
                # the server's name had not, which is a ratio of 0 rather than
                # an undefined one — the same call the un-stripped reading makes
                # for a tool whose words are all shared.
                if not stripped:
                    domain_stripped.append(0.0)
                    continue
                domain_stripped.append(
                    sum(1 for w in stripped if stripped_df[w] == 1) / len(stripped))

            # ── D: the candidate collision list, emitted directly ───────────
            found = clusters(names, token_sets, domain)
            if found and len(examples["clusters"]) < 25:
                examples["clusters"].append({
                    "server": s["qualifiedName"],
                    "tools": len(tools),
                    "domain": sorted(surfaces(domain))[:8],
                    # Tightest first: a noun on two tools of ten localises the
                    # problem, a noun on nine of ten barely narrows it.
                    "tightest": [{"noun": surfaces([word])[0], "shared_by": tools_sharing}
                                 for word, tools_sharing in found[:5]],
                })

            # ── names, with any server-wide prefix taken off first ─────────
            # The prefix discriminates nothing *within* the server — every tool
            # has it — so leaving it in would drag every name down by a
            # constant and make a well-named server look collided.
            prefix = common_prefix(names)
            servers_with_prefix += prefix is not None
            name_sets = []
            for n in names:
                words = name_words(n)
                if prefix is not None and words and words[0] == prefix:
                    words = words[1:]
                # After the prefix comes off, so the verb is genuinely first:
                # on `Gmail_GetThread` the leading token is `gmail` until it is
                # stripped, and the table would never see `get`.
                name_sets.append(set(canonical_verb(words)))

            zero_here = False
            desc_ratios = []
            desc_df = document_frequency(token_sets)
            for i, t in enumerate(tools):
                mine = token_sets[i]
                if not mine:
                    desc_ratios.append(None)
                    continue  # an undescribed tool has no ratio, not a ratio of 0
                # Used by exactly one tool, which is this one.
                unique = {w for w in mine if desc_df[w] == 1}
                ratio = len(unique) / len(mine)
                distinctive.append(ratio)
                desc_ratios.append(ratio)
                if not unique:
                    zero_here = True
                    if len(examples["zero_distinctive"]) < 25:
                        examples["zero_distinctive"].append({
                            "server": s["qualifiedName"],
                            "tool": t.get("name"),
                            "tokens": len(mine),
                            "description": (t.get("description") or "")[:200],
                        })
            servers_with_zero_distinctive += zero_here

            # ── the same ratio over names, and the cross of the two ────────
            name_df = document_frequency(name_sets)
            for i, t in enumerate(tools):
                mine = name_sets[i]
                if not mine:
                    continue
                name_ratio = sum(1 for w in mine if name_df[w] == 1) / len(mine)
                name_distinctive.append(name_ratio)

                if desc_ratios[i] is None:
                    continue
                low_desc = desc_ratios[i] < LOW_DISTINCTIVE
                low_name = name_ratio < LOW_DISTINCTIVE
                # The four quadrants. `both_low` is the serious one: if the name
                # is what a model falls back to when descriptions collide, a
                # tool in there has left it nothing at all.
                quadrant["both_low" if low_desc and low_name else
                         "name_carries" if low_desc else
                         "desc_carries" if low_name else "both_clear"] += 1

                if low_desc and low_name and len(examples["both_low"]) < 25:
                    examples["both_low"].append({
                        "server": s["qualifiedName"],
                        "tool": t.get("name"),
                        "description_share": round(desc_ratios[i], 3),
                        "name_share": round(name_ratio, 3),
                        "description": (t.get("description") or "")[:160],
                    })

            # ── input overlap, per pair ───────────────────────────────────
            # Capped, because this is O(n²) and one 275-tool server is 37,675
            # pairs on its own — which would also dominate any pair-weighted
            # figure, the same reason the cosine curve leads with servers.
            if len(tools) <= 60:
                nouns = [input_nouns(t) for t in tools]
                outputs = [output_terms(t) for t in tools]
                flagged_here = healthy_here = near_here = False
                for i in range(len(tools)):
                    for j in range(i + 1, len(tools)):
                        overlap_pairs += 1
                        overlap = jaccard(nouns[i], nouns[j])
                        if overlap < INPUT_OVERLAP_FLAG:
                            continue
                        overlap_pairs_sharing_inputs += 1

                        # ── C: the gate ──────────────────────────────────
                        # Sharing an input is not the failure. Sharing an input
                        # while neither tool states an output a reader could
                        # tell from the other's is the failure. Without this,
                        # every well-designed pair that takes the same argument
                        # and returns different shapes is a false positive — and
                        # on a large server a 10% false-positive rate buries the
                        # real signal completely.
                        if outputs_distinguishable(outputs[i], outputs[j]):
                            overlap_pairs_near_miss += 1
                            near_here = True
                            if len(examples["output_near_miss"]) < 25:
                                examples["output_near_miss"].append({
                                    "server": s["qualifiedName"],
                                    "a": tools[i].get("name"),
                                    "b": tools[j].get("name"),
                                    "input_overlap": round(overlap, 3),
                                    "shared_inputs": sorted(surfaces(nouns[i] & nouns[j]))[:10],
                                    "a_output": sorted(surfaces(outputs[i]))[:8],
                                    "b_output": sorted(surfaces(outputs[j]))[:8],
                                })
                            continue

                        overlap_pairs_flagged += 1
                        flagged_here = True

                        # Both descriptions cleared the corpus median and this
                        # still fired: the dangerous combination, because the
                        # existing metric has already called the pair healthy.
                        if (desc_ratios[i] is not None and desc_ratios[j] is not None
                                and desc_ratios[i] >= CORPUS_MEDIAN_DISTINCTIVE
                                and desc_ratios[j] >= CORPUS_MEDIAN_DISTINCTIVE):
                            overlap_pairs_looks_healthy += 1
                            healthy_here = True
                            if len(examples["input_overlap"]) < 25:
                                examples["input_overlap"].append({
                                    "server": s["qualifiedName"],
                                    "a": tools[i].get("name"),
                                    "b": tools[j].get("name"),
                                    "overlap": round(overlap, 3),
                                    "shared": sorted(surfaces(nouns[i] & nouns[j]))[:10],
                                    "a_distinctive": round(desc_ratios[i], 3),
                                    "b_distinctive": round(desc_ratios[j], 3),
                                    # Empty on both sides is the common case and
                                    # the reason the pair is flagged at all.
                                    "a_output": sorted(surfaces(outputs[i]))[:8],
                                    "b_output": sorted(surfaces(outputs[j]))[:8],
                                })
                servers_with_overlap += flagged_here
                servers_with_near_miss += near_here
                servers_with_looks_healthy += healthy_here

                # ── D: the gap between the two readings ──────────────────
                # Reading 1 says the descriptions do not look alike once the
                # server's own name is taken out. Reading 2 says two of its
                # tools could still both be right for one sentence. A server
                # that is clean on the first and flagged on the second is the
                # shape neither metric describes on its own, and averaging them
                # would report it as unremarkable.
                if flagged_here:
                    clean = [r for r in domain_stripped[-len(tools):] if r is not None]
                    if clean and median(sorted(clean)) >= CORPUS_MEDIAN_DISTINCTIVE:
                        servers_in_the_gap += 1
                        if len(examples["reading_gap"]) < 25:
                            examples["reading_gap"].append({
                                "server": s["qualifiedName"],
                                "tools": len(tools),
                                "domain_stripped_median": round(median(sorted(clean)), 3),
                                "shares_an_object": True,
                            })

        # pairwise description overlap, within this server only
        best = 0.0
        best_pair = None
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                total_pairs += 1
                sim = cosine(vectors[i], vectors[j])
                for th in thresholds:
                    if sim >= th:
                        pairs_over[th] += 1
                if sim > best:
                    best, best_pair = sim, (tools[i], tools[j])
        for th in thresholds:
            if best >= th:
                servers_over[th] += 1
        if best >= 0.8 and best_pair and len(examples["overlap"]) < 25:
            examples["overlap"].append({
                "server": s["qualifiedName"],
                "similarity": round(best, 3),
                "a": {"name": best_pair[0].get("name"),
                      "description": (best_pair[0].get("description") or "")[:200]},
                "b": {"name": best_pair[1].get("name"),
                      "description": (best_pair[1].get("description") or "")[:200]},
            })

        servers_with_undescribed_param += s_undesc_param
        servers_with_undescribed_tool += s_undesc_tool
        servers_with_stated_unconstrained += s_stated
        servers_with_restated_param += s_restated_param
        servers_with_nested_gap += s_nested_gap
        servers_with_ambiguous_branch += s_ambiguous_branch

    def pct(a, b):
        return round(100 * a / b, 1) if b else None

    tps = sorted(tools_per_server)
    ppt = sorted(params_per_tool)
    sbs = sorted(schema_bytes_per_server)

    return {
        "n_servers": n_servers,
        "n_tools": n_tools,
        "n_params": n_params,
        "schema_bytes": {
            "median": median(sbs),
            "p90": pctile(sbs, 0.9),
            "max": sbs[-1] if sbs else None,
            # bytes/4 is the same approximation MCPulse's cost model uses.
            "median_tokens_approx": round(median(sbs) / 4) if sbs else None,
            "p90_tokens_approx": round(pctile(sbs, 0.9) / 4) if sbs else None,
        },
        "tools_per_server": {
            "median": median(tps),
            "mean": round(sum(tps) / len(tps), 1) if tps else None,
            "max": tps[-1] if tps else None,
            "over_10": pct(sum(1 for x in tps if x > 10), len(tps)),
            "over_20": pct(sum(1 for x in tps if x > 20), len(tps)),
        },
        "params_per_tool": {
            "median": median(ppt),
            "mean": round(sum(ppt) / len(ppt), 1) if ppt else None,
            "max": ppt[-1] if ppt else None,
            "over_6": pct(sum(1 for x in ppt if x > 6), len(ppt)),
            "zero": pct(sum(1 for x in ppt if x == 0), len(ppt)),
        },
        "undescribed_tools": {
            "tools_pct": pct(undescribed_tools, n_tools),
            "servers_pct": pct(servers_with_undescribed_tool, n_servers),
            "count": undescribed_tools,
        },
        # Absence only. This is the definition the post published and it is
        # left alone on purpose, so the figure stays comparable with what has
        # already gone out. It is a floor: see `restated_params` below.
        "undescribed_params": {
            "params_pct": pct(undescribed_params, n_params),
            "servers_pct": pct(servers_with_undescribed_param, n_servers),
            "count": undescribed_params,
        },
        # Populated and empty of information. The commoner failure, and the
        # one that passes every linter.
        "restated_params": {
            "params_pct": pct(restated_params, n_params),
            "servers_pct": pct(servers_with_restated_param, n_servers),
            "count": restated_params,
        },
        # The two together — what a model actually has to guess at, and the
        # number to read now that both are measured.
        "substantively_undescribed_params": {
            "params_pct": pct(undescribed_params + restated_params, n_params),
            "count": undescribed_params + restated_params,
        },
        # ── H: what the schema costs to fill in, split from how deep it is ──
        #
        # `leaves` is the size of the filling-in job. `structural_depth` is the
        # existing max over the tree. `decision_depth` is how many of those
        # levels are a branch the model has to choose at — the part depth cannot
        # see, and the part it can get wrong.
        "schema_shape": {
            "leaves": {
                "median": median(sorted(leaf_counts)),
                "p90": pctile(sorted(leaf_counts), 0.9),
                "max": max(leaf_counts) if leaf_counts else None,
            },
            "structural_depth": {
                "median": median(sorted(structural_depths)),
                "p90": pctile(sorted(structural_depths), 0.9),
                "max": max(structural_depths) if structural_depths else None,
            },
            "decision_depth": {
                "median": median(sorted(decision_depths)),
                "p90": pctile(sorted(decision_depths), 0.9),
                "max": max(decision_depths) if decision_depths else None,
                # The contrast worth publishing: almost every schema is deep
                # without being a decision.
                "tools_with_any_pct": pct(sum(1 for d in decision_depths if d > 0),
                                          len(decision_depths)),
            },
            "branches": {
                "tools_with_a_branch_pct": pct(tools_with_branches, n_tools),
                "branch_points": branch_points_total,
                # Branches that share most of their properties and carry no
                # discriminator — a choice with nothing to make it on.
                "tools_with_an_ambiguous_branch": ambiguous_branches,
                "tools_with_an_ambiguous_branch_pct": pct(ambiguous_branches, n_tools),
                "servers_pct": pct(servers_with_ambiguous_branch, n_servers),
                "note": (
                    "Scored on property-set overlap and the presence of a "
                    "discriminator. Required-set overlap would be the stronger "
                    "term and is not available: the registry strips the "
                    "top-level `required` array. See excluded_metrics."
                ),
            },
        },
        # ── F: every depth ──────────────────────────────────────────────────
        # A different denominator, so not comparable with the figures above and
        # deliberately not merged with them. A model cannot see the difference
        # between an undescribed field at depth one and one at depth four; the
        # published rate can see only the first.
        "deep_params": {
            "params_total": deep_params_total,
            "nesting_multiplier": (round(deep_params_total / n_params, 2)
                                   if n_params else None),
            "undescribed_pct": pct(deep_undescribed, deep_params_total),
            "restated_pct": pct(deep_restated, deep_params_total),
            "substantively_undescribed_pct": pct(deep_undescribed + deep_restated,
                                                 deep_params_total),
            # The population the published definition is blind to: every
            # top-level parameter described, something below it bare.
            "tools_clean_on_top_bare_below": tools_clean_on_top_bare_below,
            "tools_clean_on_top_bare_below_pct": pct(tools_clean_on_top_bare_below, n_tools),
            "servers_pct": pct(servers_with_nested_gap, n_servers),
        },
        "enums": {
            "string_params": string_params,
            "enum_params": enum_params,
            "enum_pct_of_all_params": pct(enum_params, n_params),
            "stated_unconstrained": stated_unconstrained,
            "stated_unconstrained_pct_of_strings": pct(stated_unconstrained, string_params),
            "servers_pct": pct(servers_with_stated_unconstrained, n_servers),
        },
        "distinctive_ratio": {
            "tools_measured": len(distinctive),
            "median": round(median(sorted(distinctive)), 3) if distinctive else None,
            "under_20pct": pct(sum(1 for x in distinctive if x < 0.2), len(distinctive)),
            "under_10pct": pct(sum(1 for x in distinctive if x < 0.1), len(distinctive)),
            "exactly_zero": pct(sum(1 for x in distinctive if x == 0), len(distinctive)),
            "servers_with_a_zero_pct": pct(servers_with_zero_distinctive, n_servers),
        },
        # Whether two tools could both be correct for one request, which is a
        # different question from whether their descriptions look alike.
        "input_overlap": {
            "pairs_measured": overlap_pairs,
            # Before the output gate. Kept so the two numbers reconcile: this
            # minus `near_miss_pairs` is `flagged_pairs`, and a reader can see
            # exactly how much work the gate is doing rather than taking it on
            # trust.
            "sharing_inputs_pairs": overlap_pairs_sharing_inputs,
            "sharing_inputs_pct": pct(overlap_pairs_sharing_inputs, overlap_pairs),
            # After it. Shares an object *and* neither tool states an output the
            # other's could be told from.
            "flagged_pairs": overlap_pairs_flagged,
            "flagged_pairs_pct": pct(overlap_pairs_flagged, overlap_pairs),
            "servers_pct": pct(servers_with_overlap, n_servers),
            # Shares an object and states different outputs. Reported without
            # alarm: this is what good design looks like when two tools take the
            # same argument, and the six-tool ticket server that motivated the
            # gate lives here rather than in the flags.
            "near_miss_pairs": overlap_pairs_near_miss,
            "near_miss_pct": pct(overlap_pairs_near_miss, overlap_pairs),
            "near_miss_servers_pct": pct(servers_with_near_miss, n_servers),
            # How much of the raw signal was false positives.
            "gate_removed_pct": pct(overlap_pairs_near_miss, overlap_pairs_sharing_inputs),
            # The dangerous combination: both descriptions cleared the corpus
            # median and the pair still shares its object, so the existing
            # metric has already called it healthy and nothing else will
            # mention it.
            "looks_healthy_pairs": overlap_pairs_looks_healthy,
            "looks_healthy_servers_pct": pct(servers_with_looks_healthy, n_servers),
        },
        # The same distinctive measure over tool names — the backstop when a
        # description stops discriminating.
        "name_distinctive_ratio": {
            "tools_measured": len(name_distinctive),
            "median": (round(median(sorted(name_distinctive)), 3)
                       if name_distinctive else None),
            "under_10pct": pct(sum(1 for x in name_distinctive if x < 0.1),
                               len(name_distinctive)),
            "exactly_zero": pct(sum(1 for x in name_distinctive if x == 0),
                                len(name_distinctive)),
            "servers_with_a_prefix_pct": pct(servers_with_prefix, n_servers),
            # The cross of description and name. `both_low` is the publishable
            # one: nothing left to discriminate on, and nobody has measured it.
            "quadrants": {
                name: pct(count, sum(quadrant.values()))
                for name, count in sorted(quadrant.items())
            },
            "both_low_count": quadrant["both_low"],
        },
        # ── D: two readings of the same server, never averaged ───────────
        #
        #   reading 1  domain-stripped distinctiveness — do these descriptions
        #              look alike once the server's own name is taken out?
        #              Stops penalising focused servers.
        #   reading 2  domain-intact input overlap — could two of these tools be
        #              right for one sentence? Needs the domain noun, because on
        #              a focused server the domain noun *is* the collision.
        #
        # `in_the_gap_servers_pct` is the finding: clean on the first, flagged
        # on the second. A focused server whose tools all act on one object and
        # never say what they return.
        "domain_and_clusters": {
            "servers_with_a_domain_noun_pct": pct(servers_with_domain, n_servers),
            "domain_nouns_per_such_server": (round(domain_noun_count / servers_with_domain, 1)
                                             if servers_with_domain else None),
            "reading_1_domain_stripped": {
                "tools_measured": len(domain_stripped),
                "median": (round(median(sorted(domain_stripped)), 3)
                           if domain_stripped else None),
                "exactly_zero": pct(sum(1 for x in domain_stripped if x == 0),
                                    len(domain_stripped)),
                "under_10pct": pct(sum(1 for x in domain_stripped if x < 0.1),
                                   len(domain_stripped)),
            },
            "in_the_gap_servers_pct": pct(servers_in_the_gap, n_servers),
            "in_the_gap_servers": servers_in_the_gap,
        },
        "overlap": {
            "total_pairs": total_pairs,
            # Lead with the server figure. Pair counts are dominated by a
            # handful of enormous servers — one in the sample publishes 275
            # tools, which is 37,675 pairs on its own, so a pair-weighted
            # percentage would mostly describe that server.
            "servers_pct": {str(t): pct(servers_over[t], n_servers) for t in thresholds},
            "pairs_pct": {str(t): pct(pairs_over[t], total_pairs) for t in thresholds},
        },
        "duplicate_tool_names_servers_pct": pct(servers_with_dup_names, n_servers),
    }


def main():
    rows = load()

    scanned = [r for r in rows if r.get("tools")]
    null_tools = [r for r in rows if r.get("tools") is None and not r.get("detail_failed")]
    failed = [r for r in rows if r.get("detail_failed")]
    empty = [r for r in rows if r.get("tools") == []]

    print(f"frame rows      : {len(rows)}")
    print(f"  detail failed : {len(failed)}")
    print(f"  tools == null : {len(null_tools)}   (registry never scanned it)")
    print(f"  tools == []   : {len(empty)}")
    print(f"  usable        : {len(scanned)}")

    # idf over every tool description in the corpus
    df = Counter()
    n_docs = 0
    for r in scanned:
        for t in r["tools"]:
            n_docs += 1
            df.update(set(tokenize(t.get("description") or "")))
    idf = {w: math.log((1 + n_docs) / (1 + c)) + 1 for w, c in df.items()}

    out = {
        "corpus": {
            "frame": len(rows),
            "detail_failed": len(failed),
            "never_scanned": len(null_tools),
            "zero_tools": len(empty),
            "usable_servers": len(scanned),
        },
        "excluded_metrics": {
            "required_vs_optional":
                "The registry drops the top-level `required` array. Ground truth "
                "from four reference servers booted locally: 27 of 37 tools emit "
                "one, the registry reports it for none. Not measurable here.",
            # ── The second stripped field, found the same way ────────────────
            # The plan for this check assumed annotations ride along in the
            # registry payload, so it would need no new collection. They do not.
            # The corpus carries the field for not one of 82,549 tools, and the
            # same four reference servers emit it for all 37 of theirs, with
            # every one of the four hints represented.
            #
            # So "0% of public servers set destructiveHint" would have been a
            # fact about Smithery published as a fact about MCP authors — the
            # exact mistake `required` already caught once.
            "tool_annotations":
                "The registry drops `annotations`. Ground truth from four "
                "reference servers booted locally: 37 of 37 tools emit one, "
                "carrying all four hints (readOnlyHint, destructiveHint, "
                "idempotentHint, openWorldHint); the registry reports it for "
                "none of 82,549. The prose-versus-annotation checks therefore "
                "run in the checker, where an author pastes their own "
                "tools/list, and no corpus figure is published for them.",
        },
        "pools": {},
    }
    examples = defaultdict(list)

    for pool in ("popular", "longtail"):
        subset = [r for r in scanned if r.get("pool") == pool]
        if subset:
            print(f"\nanalysing {pool}: {len(subset)} servers")
            out["pools"][pool] = analyse_pool(subset, idf, examples)

    print("\nanalysing all")
    out["pools"]["all"] = analyse_pool(scanned, idf, examples)

    # ── by server size ────────────────────────────────────────────────────
    # The most useful cut in the study, and the one that separates the two
    # failures: collision scales hard with tool count and missing parameter
    # descriptions do not move at all.
    print("\nanalysing by size")
    buckets = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 60), (61, 10 ** 9)]
    by_size = []
    for lo, hi in buckets:
        subset = [r for r in scanned if lo <= len(r["tools"]) <= hi]
        if not subset:
            continue
        stats = analyse_pool(subset, idf, defaultdict(list))
        by_size.append({
            "tools_per_server": f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+",
            "servers": stats["n_servers"],
            "params": stats["n_params"],
            "undescribed_params_pct": stats["undescribed_params"]["params_pct"],
            "restated_params_pct": stats["restated_params"]["params_pct"],
            "zero_distinctive_pct": stats["distinctive_ratio"]["exactly_zero"],
            "zero_name_distinctive_pct": stats["name_distinctive_ratio"]["exactly_zero"],
            "both_low_pct": stats["name_distinctive_ratio"]["quadrants"].get("both_low"),
            "input_overlap_pairs_pct": stats["input_overlap"]["flagged_pairs_pct"],
        })
    out["by_server_size"] = by_size

    with open(os.path.join(DATA, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(DATA, "examples.json"), "w") as f:
        json.dump(dict(examples), f, indent=1)

    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
