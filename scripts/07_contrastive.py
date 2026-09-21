"""
Ask a model to choose, and keep every mistake.

── The ceiling this exists to get past ─────────────────────────────────────
Two tools can share no content word — after stemming, after every correction in
this repo — and still collide, because one user sentence names both their
objects and the tools partition an intent the sentence does not.

That is not hypothetical. On the job-applications server this study keeps
returning to, `analyze_job_description` and `score_resume` have **zero** lexical
overlap and their author reports they are confusable in practice. "Check my
resume for this job" is a fair sentence for both. `05_fixtures.py` pins that pair
as a non-flag, because nothing computed on descriptions alone reaches it.

A labelled set of intent-to-tool pairs does. That is what this builds.

── What it produces, and why the artefact is the point ─────────────────────
Every mistaken selection becomes a fixture: prompt, expected tool, actual tool.
Stored per server. That turns a description rewrite from hopeful into testable —
the next person who "improves" the wording finds out immediately whether they
broke the distinction.

It is the same loop MCPulse's before/after view measures in production, run
before shipping instead of after.

── And the study it makes possible ─────────────────────────────────────────
Run across the corpus, every static signal in this repo — distinctive share,
input overlap, name distinctiveness, branch ambiguity — can be *ranked* by how
well it predicts a real mistake, rather than all being reported as equally
meaningful. That is the study that earns the word "misuse", and it is what says
which of the checks to put first in the checker.

    export ANTHROPIC_API_KEY=...
    python3 scripts/07_contrastive.py --server firstexhotic/ai-applyd
    python3 scripts/07_contrastive.py --corpus --limit 200

Writes `data/contrastive.json`: every prompt, what was expected, what was picked.
"""

import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))

from importlib.machinery import SourceFileLoader  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
A = SourceFileLoader("analyse", os.path.join(HERE, "03_analyse.py")).load_module()

#: The model that generates the prompts and the models that choose.
#:
#: Two choosers, because the file this came from asks for at least two: if the
#: failures diverge between them, that is a significant finding in itself about
#: writing for a specific client rather than for "a model".
AUTHOR_MODEL = "claude-opus-5"
CHOOSER_MODELS = ["claude-opus-5", "claude-sonnet-5"]

API = "https://api.anthropic.com/v1/messages"


def call(model, system, messages, tools=None, max_tokens=4096, effort="low"):
    """One request. Raises on anything that is not a 200, because a silent
    failure here would look like a model choosing badly.

    ── Two settings that are correctness, not tuning ─────────────────────────
    **`max_tokens` is generous.** Thinking is on by default on these models and
    its tokens count against this ceiling. A tight cap can exhaust the budget
    during reasoning and return no `tool_use` block at all — which this harness
    would record as "the model picked nothing", a false result that looks
    exactly like a real one.

    **Thinking stays on.** Disabling it is the obvious way to make a
    pick-one-tool task cheap, and it introduces the one failure this experiment
    cannot survive: with thinking off, the model sometimes writes the tool call
    into its *visible text* instead of emitting a `tool_use` block. The turn
    succeeds, nothing is recorded, and every such case would be scored as
    "picked neither". Low effort gets the cost back without that risk.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "output_config": {"effort": effort},
    }
    if tools:
        body["tools"] = tools

    request = urllib.request.Request(
        API,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{error.code} from the API: {error.read()[:400]!r}") from error


def candidate_pairs(tools):
    """Pairs worth asking about: the ones the static signals flag.

    Deliberately not every pair. The point is to test where the signals say
    there is a problem — and, just as importantly, where they say there is not:
    a pair the signals cleared that a model still gets wrong is the finding.
    """
    nouns = [A.input_nouns(t) for t in tools]
    outputs = [A.output_terms(t) for t in tools]
    names = [t.get("name") for t in tools]
    token_sets = [set(A.tokenize(t.get("description") or "")) for t in tools]
    df = A.document_frequency(token_sets)

    out = []
    for i in range(len(tools)):
        for j in range(i + 1, len(tools)):
            overlap = A.jaccard(nouns[i], nouns[j])
            distinguishable = A.outputs_distinguishable(outputs[i], outputs[j])
            shared_words = {w for w in (token_sets[i] & token_sets[j]) if df[w] > 1}

            out.append({
                "a": names[i],
                "b": names[j],
                # Every signal, recorded per pair. This is the column the
                # ranking study needs: which of them predicted the mistake.
                "input_overlap": round(overlap, 3),
                "outputs_distinguishable": distinguishable,
                "flagged_by_input_overlap": overlap >= A.INPUT_OVERLAP_FLAG and not distinguishable,
                "shared_content_words": sorted(A.surfaces(shared_words))[:8],
                "name_overlap": round(
                    A.jaccard(set(A.name_words(names[i])), set(A.name_words(names[j]))), 3
                ),
            })

    return out


PROMPT_SYSTEM = """\
You write realistic user requests for testing MCP tool selection.

Given two tools from one server, write ONE sentence a user would plausibly say \
to an assistant that makes BOTH tools a defensible choice. Name the shared \
object without resolving which output is wanted.

Do not mention tool names. Do not hedge or offer alternatives. Write the \
sentence a real person would type, and nothing else."""


def write_prompt(pair, tools_by_name):
    a = tools_by_name[pair["a"]]
    b = tools_by_name[pair["b"]]

    reply = call(
        AUTHOR_MODEL,
        PROMPT_SYSTEM,
        [{
            "role": "user",
            "content": (
                f"Tool A — {a['name']}: {a.get('description') or '(no description)'}\n"
                f"Tool B — {b['name']}: {b.get('description') or '(no description)'}"
            ),
        }],
        max_tokens=1024,
    )

    for block in reply.get("content", []):
        if block.get("type") == "text":
            return block["text"].strip().strip('"')
    return None


def as_api_tools(tools):
    """The server's tools in the shape the Messages API takes.

    The whole list goes over, not just the pair: a model choosing between two
    tools in isolation is not the situation any of this is about.
    """
    out = []
    for tool in tools:
        schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
        if not isinstance(schema, dict) or schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        out.append({
            "name": tool["name"],
            "description": (tool.get("description") or "")[:1024],
            "input_schema": schema,
        })
    return out


def choose(model, api_tools, prompt):
    """What the model picks, or None when it picks nothing."""
    reply = call(
        model,
        "You are an assistant with tools. Use one if it fits the request.",
        [{"role": "user", "content": prompt}],
        tools=api_tools,
    )

    for block in reply.get("content", []):
        if block.get("type") == "tool_use":
            return block["name"]
    return None


def run_server(name, tools, per_server):
    api_tools = as_api_tools(tools)
    by_name = {t["name"]: t for t in tools}

    pairs = candidate_pairs(tools)
    # Flagged pairs first — those are the ones the signals claim are a problem —
    # then a sample of the rest, which is where a signal can be shown to have
    # missed something.
    pairs.sort(key=lambda p: (not p["flagged_by_input_overlap"], -p["input_overlap"]))
    chosen = pairs[:per_server]

    cases = []
    for pair in chosen:
        prompt = write_prompt(pair, by_name)
        if not prompt:
            continue

        picked = {model: choose(model, api_tools, prompt) for model in CHOOSER_MODELS}

        # There is no single right answer here, and pretending otherwise is the
        # trap. The prompt was written to make both defensible, so what is
        # recorded is which tool each model reached for — and whether they
        # agreed with each other.
        agreed = len(set(picked.values())) == 1
        cases.append({
            "server": name,
            "pair": [pair["a"], pair["b"]],
            "prompt": prompt,
            "picked": picked,
            "models_agree": agreed,
            # Neither model picked either tool: the pair is not the problem, the
            # prompt is, and the case is kept rather than quietly dropped.
            "picked_neither": all(
                value not in (pair["a"], pair["b"]) for value in picked.values()
            ),
            "signals": {
                key: pair[key]
                for key in (
                    "input_overlap",
                    "outputs_distinguishable",
                    "flagged_by_input_overlap",
                    "shared_content_words",
                    "name_overlap",
                )
            },
        })
        print(f"  {pair['a']} / {pair['b']}: {picked}", flush=True)

    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", help="one qualifiedName from the corpus")
    parser.add_argument("--corpus", action="store_true", help="sample across the corpus")
    parser.add_argument("--limit", type=int, default=20, help="servers, with --corpus")
    parser.add_argument("--pairs", type=int, default=5, help="pairs per server")
    parser.add_argument("--seed", type=int, default=1, help="so a sample is reproducible")
    parser.add_argument(
        "--max-tools",
        type=int,
        default=60,
        help=(
            "skip servers with more tools than this. The whole tool list goes into "
            "every chooser call, so one 2,530-tool server costs more than a hundred "
            "ordinary ones — and the same cap is already on the pair analysis in 03."
        ),
    )
    args = parser.parse_args()

    fixtures = os.path.join(DATA, "fixture_servers.json")
    servers = {}

    if args.server and os.path.exists(fixtures):
        held = json.load(open(fixtures))
        if args.server in held:
            servers[args.server] = held[args.server]

    if not servers:
        rows = A.load()
        usable = [
            r for r in rows
            if 2 <= len(r.get("tools") or []) <= args.max_tools
        ]
        if args.server:
            servers = {
                r["qualifiedName"]: r["tools"]
                for r in usable
                if r["qualifiedName"] == args.server
            }
            if not servers:
                raise SystemExit(f"{args.server} is not in the corpus")
        else:
            random.Random(args.seed).shuffle(usable)
            servers = {r["qualifiedName"]: r["tools"] for r in usable[: args.limit]}

    cases = []
    for name, tools in servers.items():
        print(f"\n{name}: {len(tools)} tools", flush=True)
        cases.extend(run_server(name, tools, args.pairs))

    out = {
        "author_model": AUTHOR_MODEL,
        "chooser_models": CHOOSER_MODELS,
        "servers": len(servers),
        "cases": cases,
        # The ranking the whole exercise is for: of the pairs a model treated as
        # interchangeable, how many did each static signal flag? A signal that
        # flags none of them is measuring something else.
        "signal_recall": signal_recall(cases),
    }

    path = os.path.join(DATA, "contrastive.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n{len(cases)} cases across {len(servers)} servers -> data/contrastive.json")
    for signal, recall in out["signal_recall"].items():
        print(f"  {signal:34} {recall}")


def signal_recall(cases):
    """How often each signal flagged a pair the models actually split on.

    A pair where the two models chose differently is a pair a model could go
    either way on — which is as close to ground truth as this gets without
    asking the server's author. The question for each signal is how many of
    those it saw coming.
    """
    split = [c for c in cases if not c["models_agree"] and not c["picked_neither"]]
    if not split:
        return {"pairs_models_split_on": 0}

    def share(predicate):
        return round(100 * sum(1 for c in split if predicate(c)) / len(split), 1)

    return {
        "pairs_models_split_on": len(split),
        "input_overlap_flagged_pct": share(lambda c: c["signals"]["flagged_by_input_overlap"]),
        "shared_a_content_word_pct": share(lambda c: bool(c["signals"]["shared_content_words"])),
        "outputs_indistinguishable_pct": share(
            lambda c: not c["signals"]["outputs_distinguishable"]
        ),
        "names_overlapped_pct": share(lambda c: c["signals"]["name_overlap"] >= 0.5),
    }


if __name__ == "__main__":
    main()
