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

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "from", "at", "as", "is", "are", "be", "this", "that", "it", "its", "you",
    "your", "will", "can", "if", "when", "use", "used", "using", "returns",
    "return", "tool", "get", "set", "list", "all", "any", "not", "no", "only",
}

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


def tokenize(text):
    return [w for w in re.findall(r"[a-z][a-z0-9_]+", (text or "").lower())
            if w not in STOPWORDS and len(w) > 2]


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
                if not pdesc:
                    undescribed_params += 1
                    s_undesc_param = True
                    if len(examples["undescribed_param"]) < 25:
                        examples["undescribed_param"].append(
                            {"server": s["qualifiedName"], "tool": t.get("name"),
                             "param": pname})

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

        # ── distinctive ratio, per tool, against its own siblings ──────────
        # Only meaningful with something to be distinct *from*, so
        # single-tool servers are skipped rather than scored 1.0.
        if len(tools) >= 2:
            token_sets = [set(tokenize(t.get("description") or "")) for t in tools]
            zero_here = False
            for i, t in enumerate(tools):
                mine = token_sets[i]
                if not mine:
                    continue  # an undescribed tool has no ratio, not a ratio of 0
                others = set().union(*[token_sets[j]
                                       for j in range(len(tools)) if j != i])
                unique = mine - others
                ratio = len(unique) / len(mine)
                distinctive.append(ratio)
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

    def pct(a, b):
        return round(100 * a / b, 1) if b else None

    tps = sorted(tools_per_server)
    ppt = sorted(params_per_tool)
    sbs = sorted(schema_bytes_per_server)

    def median(xs):
        return xs[len(xs) // 2] if xs else None

    def pctile(xs, q):
        return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None

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
        "undescribed_params": {
            "params_pct": pct(undescribed_params, n_params),
            "servers_pct": pct(servers_with_undescribed_param, n_servers),
            "count": undescribed_params,
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
            "zero_distinctive_pct": stats["distinctive_ratio"]["exactly_zero"],
        })
    out["by_server_size"] = by_size

    with open(os.path.join(DATA, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(DATA, "examples.json"), "w") as f:
        json.dump(dict(examples), f, indent=1)

    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
