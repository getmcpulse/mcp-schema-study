# mcp-schema-study

What models are actually handed when they connect to a public MCP server.

This is the data and code behind [We read the schemas of 4,951 public MCP
servers](https://getmcpulse.com/blog/reading-5000-mcp-schemas).
Everything below is reproducible with Python 3.12, `httpx`, and about twenty
minutes.

**[getmcpulse.com](https://getmcpulse.com)** · [Docs](https://docs.getmcpulse.com) · [Dashboard](https://app.getmcpulse.com)

```bash
python3 scripts/01_collect.py               # build the corpus
python3 scripts/02_validate_ground_truth.py # check the corpus is trustworthy
python3 scripts/03_analyse.py               # every figure in the post
python3 scripts/04_stem_fixture.py          # the stemmer fixture the checker shares
python3 scripts/05_fixtures.py              # the cases every signal must get right
python3 scripts/06_extract_fixture.py       # pull the named servers out of the corpus
python3 scripts/07_contrastive.py --help    # ask a model to choose (needs an API key)
```

> ## The published figures were wrong, and these are the corrected ones
>
> Every word-counting metric here did exact string matching, so `screens` and
> `screening` were two different words. Two tools sharing their most meaningful
> term both scored it as distinctive.
>
> | | published | corrected |
> |---|---|---|
> | zero-distinctive tools | 17.7% | **22.9%** |
> | median distinctive share | 26% | **20%** |
> | under 10% distinctive | 27.0% | **34.1%** |
> | nothing distinct in description *or* name | 20.9% | **26.1%** |
> | 61+ tool servers | 32.4% | **40.8%** |
>
> The parameter rates did not move, which is the check that this was a
> word-counting bug and nothing broader: they test whether a field is present,
> and no stemmer can change that.
>
> Stemming can only merge words, never split them, so the direction was
> predictable and only the size was not. `scripts/stem.py` has the algorithm and
> the one deviation from Porter it makes.

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
| **`annotations`** | **37 of 37 tools** | **0** | ❌ **stripped** |

So required-versus-optional is not measured anywhere in this study. Not
because the number would be unflattering — because it would be the registry's
number rather than the servers'.

**The same check caught a second stripped field.** The plan for the
annotations work assumed `readOnlyHint` and friends ride along in the registry
payload, so it would need no new collection. They do not: the corpus carries
`annotations` for not one of 82,549 tools, and all four reference servers emit
them for all 37 of theirs, with every hint represented.

"0% of public servers set `destructiveHint`" would have been a fact about
Smithery published as a fact about MCP authors — the exact mistake `required`
already caught once. So the prose-versus-annotation checks run in the checker,
where an author pastes their own `tools/list`, and no corpus figure exists for
them.

Everything else survives intact, and every metric in `03_analyse.py` is
computed only on fields in the ✅ rows.

## Definitions

Each is a thing you can point at in the JSON, not an inference about intent.
Where a judgement call exists it is made in the direction that *under*-reports
the problem, so the figures are floors.

**undescribed parameter** — a top-level entry of `inputSchema.properties` with
no non-empty `description`. Nested object fields are not counted; a nested
field is harder to lay at the author's door and would inflate the rate.

This is **absence only**, which is what the post published, and it is a floor
rather than the rate — see the next definition.

**restates-the-name parameter** *(added after the post)* — the `description` is
populated and, once the words of the parameter's own name and a short list of
filler are removed, says nothing. `query` described as "The query", `user_id`
as "User ID", `maxResults` as "Maximum results", `filter` as "A filter to
apply".

Arguably the commoner failure, and it passes every linter as well as the
definition above. **8,195 parameters, on 15.9% of servers**, taking the real
rate from **21.5% to 24.7%**. The two are reported separately in
`analysis.json` — `undescribed_params`, `restated_params`,
`substantively_undescribed_params` — so the published figure stays the figure
that was published.

A floor in turn: the filler list is deliberately short, so one word of real
information is enough to count as a description.

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

**input overlap** *(added after the post)* — for a **pair** of tools on one
server, how much the set of things they operate on overlaps. Jaccard over each
tool's input nouns.

The ratio above asks whether two descriptions look alike. This asks whether two
tools could both be *correct* for the same request, and that is the question
that predicts a wrong selection.

The evidence is a real ten-tool server for job applications. Three of its
tools — `score_resume`, `analyze_job_description`, `optimize_resume` — score
50–60% distinctive, healthy by this corpus's own reckoning, and its author
confirms they are ambiguous in practice: "check my resume for this job" is a
fair sentence for all three. They are ambiguous because they all act on a resume
and a job posting, not because they are worded alike.

**12,169 of 564,810 pairs (2.2%) are flagged, on 25.1% of servers**, after the
output gate below removes 3,235 well-designed pairs. The subset
worth leading with is `looks_healthy_pairs`: 1,166 pairs where *both*
descriptions clear the corpus median and they still share their object, on
**8.0% of servers**. Those are the ones the distinctive ratio has already
called healthy, so nothing else in this study would ever mention them —

```
0.53  GoogleCalendar_CreateEvent / GoogleCalendar_UpdateEvent   (0.64 / 0.75 distinctive)
0.75  search_news / search_blogs                                (0.46 / 0.30)
0.61  get_weather_forecast / get_hourly_forecast                (0.33 / 0.31)
```

Input nouns are the description's content words, minus the opening verb and
minus anything after a return marker, plus every parameter name. The parameter
names do most of the work: a parameter name is the one place an author states
the object with no prose around it. No part-of-speech tagger, so adjectives
survive as "nouns" — which dilutes an overlap rather than inventing one, a floor
like everything else here.

> **An earlier version of this measured the wrong thing.** It split each
> description into a verb and the nouns it returns, and scored distinctiveness
> over the output half. Tested against that same ten-tool server it failed: all
> ten verbs were lexically distinct — score, extract, rewrite, produce, write,
> translate, return, set, apply, build — so it scored every tool at 100%,
> including the three already known to be the problem. It was replaced rather
> than extended, and none of its figures were ever published.

**output terms** *(added after the post)* — the nouns a tool says it gives back:
`outputSchema` property names, plus whatever follows a return marker. Used only
as a **gate** on input overlap.

Sharing an object is not enough. A six-tool ticket server has two tools with
identical input shapes that are correctly two tools — flat details for many
tickets against a bounded evidence bundle — and its author is right to dismiss
the flag. The rule is a conjunction:

```
overlapping inputs + distinguishable outputs   = fine design
overlapping inputs + indistinguishable outputs = the failure
```

The gate removes **3,235 of 15,404 pairs — 21% of the raw signal — as false
positives.** An unstated output counts as indistinguishable, deliberately: a tool
that never says what it returns cannot be told apart from a sibling by its
output, and that is the commonest shape of all.

**domain nouns against cluster nouns** *(added after the post)* — frequency is
not a dial, it is a distinction the count already makes. A noun on **all** of a
server's tools names the server; a noun on a **subset** names a cluster, and that
subset is the candidate collision list.

Two readings, never averaged. Stripping the domain stops penalising a focused
server — and the un-stripped reading is the one input overlap needs, because on a
focused server *the collision is the domain noun*.

**The gap is the finding.** 18.4% of servers read clean once the domain is out
and still share an object: a focused server whose tools all act on one thing and
never say what they return. Honest footnote — the complaint that drove this is
right in principle and small in practice, moving the low-distinctive rate from
34.1% to 33.6%.

**leaf count, and branch ambiguity** *(added after the post)* — depth is a max
over the tree, so one field at depth 5 and forty fields at depth 5 score
identically. Split into the size of the filling-in job and how much of it is a
choice.

The result is negative and worth publishing as such: public MCP schemas are
shallow and narrow. Median **two** leaves at depth 1; only 6.1% of tools have any
`anyOf`/`oneOf`; **38 tools in the whole corpus** have a branch whose
alternatives overlap enough to be a real choice.

**name distinctive ratio** *(added after the post)* — the distinctive ratio
again, over tool **names** rather than descriptions, with any server-wide prefix
(`gmail_`, `aiapplyd_`) stripped first. 19.3% of servers have such a prefix.

Multiple server authors report that when descriptions stop discriminating, a
model falls back to pattern-matching on the tool name. If that is so, the name
is the backstop, and a server whose descriptions collide *and* whose names
collide is far worse off than one whose names are clear.

**The theory does not survive the corpus, and that is the finding.** Names
collide *harder* than descriptions, not less: **45.1% of tool names carry no
token a sibling does not**, against 22.9% of descriptions. The quadrant where a
clear name rescues a collided description — the backstop actually working — is
the rarest of the four.

| | share of tools |
|---|---|
| both clear | 47.0% |
| description clear, name collided | 19.0% |
| **both collided** | **26.1%** |
| description collided, name clear | 8.0% |

**Better than one tool in four has nothing to discriminate on at either end** —
21,215 of them. That is `both_low`, and nobody had measured it.

**substantively undescribed, at every depth** *(added after the post)* — the
published 21.5% counts top-level `inputSchema.properties`, so a tool whose
top-level parameters are all described and whose nested object is bare scores
clean. An undescribed field at depth four is exactly as invisible to a model as
one at depth one.

Walking the whole schema finds **324,382 parameters rather than 257,287**, and
takes the substantive rate from 24.7% to **30.7%**. 1,227 tools are spotless on
top and bare below. Reported beside the published figure, never merged into it:
the denominator is different, so dividing one by the other would read as a worse
score for having a nested schema.

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

Code MIT, in [LICENSE](LICENSE). The collected data belongs to the Smithery
registry and its publishers; the committed files are derived statistics and
short excerpts, not a redistribution of the corpus.
