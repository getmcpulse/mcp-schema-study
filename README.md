# mcp-schema-study

What models are actually handed when they connect to a public MCP server.

This is the data and code behind [We read the schemas of 4,951 public MCP
servers](https://getmcpulse.com/blog/reading-5000-mcp-schemas).
Everything below is reproducible with Python 3.12, `httpx`, and about twenty
minutes.

```bash
python3 scripts/01_collect.py              # build the corpus
python3 scripts/02_validate_ground_truth.py # check the corpus is trustworthy
python3 scripts/03_analyse.py              # every figure in the post
```

Results land in `data/analysis.json`, with real instances behind each finding
in `data/examples.json` so any number here can be spot-checked against the
servers it came from.

## What this measures, and what it does not

It reads **tool schemas**. Nothing here observes a model choosing a tool,
filling an argument, or retrying — so nothing here can tell you how often
models actually get it wrong.

That distinction is the whole framing. A schema with no parameter descriptions
is a schema a model has to guess at; whether it guesses wrong, and how often,
is a question about live traffic that static analysis cannot reach. The post
says which findings are about the schema and stops there.

## Where the corpus comes from

The Smithery registry, not booted containers.

Booting each server and calling `tools/list` over stdio is the faithful method
and also a biased one: a large share of public MCP servers will not start
without real credentials, so the set that boots cleanly is not a random subset
of anything. Smithery already stores the `tools/list` response for every
server it has scanned, `inputSchema` included, which is byte-for-byte the JSON
a model receives.

The trade is stated rather than hidden: we inherit Smithery's coverage and its
scan freshness, and we learn nothing about whether a server boots.

### Two populations

`/servers` caps pagination at five pages of a hundred. Page one opens in the
tens of thousands of installs and page five ends near 1,700, so those 500 are
near enough **the most-installed servers on the registry**.

That cap is why there is a second population. `?q=` returns up to ~200 matches
per term, so a union of a few hundred terms reaches into the long tail. It is
not a random sample either — it is whatever the search ranker surfaces — but
it is a *different* bias from popularity, which is the point of having both.

| Pool | What it is |
|---|---|
| `popular` | the ~500 most-installed. Conservative: more eyes have been on these schemas. |
| `longtail` | the search union. Everything else the registry will show us. |

Every figure is reported for both. Where they disagree, the disagreement is
the finding.

## The registry normalises schemas, and one field is lost

This is the check that decides whether any of the rest is worth reading, and
it nearly sank the study.

Across the first 1,377 tools collected, **not one** carried a top-level
`required` array. Real MCP servers mark parameters required constantly, so
either the corpus was extraordinary or something in the pipeline was dropping
the field.

`02_validate_ground_truth.py` settles it by booting the four official
reference servers locally — they need no credentials and no network — taking
their real `tools/list` over stdio, and diffing field by field.

| Field | Real server | Registry | Usable? |
|---|---|---|---|
| `properties` (count) | 9 | 9 | ✅ |
| per-parameter `description` | 9 of 9 | 9 of 9 | ✅ |
| tool `description` | present | present | ✅ |
| `enum` | present | present | ✅ |
| **top-level `required`** | **27 of 37 tools** | **0** | ❌ **stripped** |

So required-versus-optional is not measured anywhere in this study. Not
because the number would be unflattering — because it would be the registry's
number rather than the servers'.

Everything else survives intact, and every metric in `03_analyse.py` is
computed only on fields in the ✅ rows.

## Definitions

Each is a thing you can point at in the JSON, not an inference about intent.
Where a judgement call exists it is made in the direction that *under*-reports
the problem, so the figures are floors.

**undescribed parameter** — a top-level entry of `inputSchema.properties` with
no non-empty `description`. Nested object fields are not counted; a nested
field is harder to lay at the author's door and would inflate the rate.

**undescribed tool** — no non-empty `description` on the tool itself.

**stated-but-unconstrained** — the parameter's description *names its valid
values* while the schema leaves it an unconstrained string with no `enum`.

This one is deliberately narrow, and the first version of it was wrong. It
matched any quoted list, which on a 40-server sample flagged 108 parameters —
things like `Filter by line (e.g., "1", "A", "F")` and `Station ID or name
(e.g., "127", "Times Square")`. Those are illustrations; a station name is not
an enum and never could be. Counting them would have produced a large,
confident, wrong number.

The test is now two-sided: the description must use explicitly closed language
(`one of`, `valid values`, `Options:`, `a | b | c`) **and** must carry no
marker that the list is illustrative (`e.g.`, `such as`, `like`, `etc.`). On
that same sample it takes 108 down to 9, and all nine are real.

A parameter whose closed set is never written down anywhere cannot be detected
from a schema at all, so this is a floor and the post says so.

**overlapping pair** — two tools in one server whose descriptions exceed a
cosine similarity on tf-idf vectors. Reported as a curve across thresholds
rather than at one number, because otherwise the threshold you pick *is* the
finding. Tools with no description have empty vectors and score zero, so they
are excluded by construction — which makes overlap a floor too, since two
undescribed tools are the most confusable pair there is.

**schema bytes** — `len(json.dumps(tools))` with `outputSchema` excluded,
since it is not always sent to the model. Needs no interpretation: it is what
the tool list costs in context before anyone asks a question.

## What is committed, and what is not

The raw corpus is roughly 35 KB per server across a few thousand servers —
a couple of hundred megabytes. It is not in the repo, and `01_collect.py`
rebuilds it from a public API in about twenty minutes.

What *is* committed is `data/analysis.json`, which holds every figure quoted in
the post, and `data/examples.json`, which holds the real servers, tools and
parameters behind each finding. So a reader can check any claim against the
thing it came from without re-running the collection — and can re-run it if
they want to check that too.

## Licence

Code MIT. The collected data belongs to the Smithery registry and its
publishers; the committed files are derived statistics and short excerpts, not
a redistribution of the corpus.
