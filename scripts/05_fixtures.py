"""
The cases every signal in this study has to get right, run on demand.

`03_analyse.py` produces percentages over 4,749 servers, and a percentage cannot
be proofread: it is either measuring what it claims to measure or it quietly is
not. The only proof is a case whose answer is known independently of the code.

Two kinds of case here, and the difference between them matters.

  **Stylised.** The one-line descriptions the write-up quotes. They are a
  paraphrase of a real server, shortened to make a mechanism visible, and they
  are what the signal was specified against. They test the mechanism.

  **Real.** The same server as the registry actually holds it, in
  `data/fixture_servers.json`, extracted from the corpus by `06_extract_fixture.py`.
  It tests the signal.

Keeping both is the point, because on this server **they disagree**, and the
disagreement is a finding rather than a bug — see the notes on `input overlap is
length-sensitive` below.

    python3 scripts/05_fixtures.py

Exits non-zero on any failure, so it can gate a commit.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from importlib.machinery import SourceFileLoader  # noqa: E402

HERE = os.path.dirname(__file__)
A = SourceFileLoader("analyse", os.path.join(HERE, "03_analyse.py")).load_module()

failures = []
checks = 0


def check(label, got, want):
    global checks
    checks += 1
    if got == want:
        print(f"pass  {label}")
    else:
        failures.append(label)
        print(f"FAIL  {label}\n        got  {got!r}\n        want {want!r}")


def note(text):
    print(f"      · {text}")


def group(name):
    print(f"\n── {name} {'─' * max(0, 62 - len(name))}")


def tool(name, description, params=(), output=None):
    out = {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object",
                        "properties": {p: {"type": "string"} for p in params}},
    }
    if output is not None:
        out["outputSchema"] = {"type": "object",
                               "properties": {p: {"type": "string"} for p in output}}
    return out


def measure(tools):
    """Distinctive share, input overlap and the output gate, for one server."""
    names = [t["name"] for t in tools]
    token_sets = [set(A.tokenize(t.get("description") or "")) for t in tools]
    nouns = [A.input_nouns(t) for t in tools]
    outputs = [A.output_terms(t) for t in tools]

    distinctive = {}
    for i, name in enumerate(names):
        if not token_sets[i]:
            continue
        others = set().union(*[token_sets[j] for j in range(len(tools)) if j != i])
        distinctive[name] = len(token_sets[i] - others) / len(token_sets[i])

    pairs = {}
    for i in range(len(tools)):
        for j in range(i + 1, len(tools)):
            overlap = A.jaccard(nouns[i], nouns[j])
            shares = overlap >= A.INPUT_OVERLAP_FLAG
            distinct_out = A.outputs_distinguishable(outputs[i], outputs[j])
            pairs[tuple(sorted((names[i], names[j])))] = {
                "overlap": round(overlap, 3),
                "shares_inputs": shares,
                "outputs_distinguishable": distinct_out,
                "flagged": shares and not distinct_out,
                "near_miss": shares and distinct_out,
            }

    domain = A.domain_nouns([s for s in token_sets if s])
    return {
        "names": names,
        "distinctive": distinctive,
        "pairs": pairs,
        "domain": domain,
        "clusters": A.clusters(names, token_sets, domain),
        "nouns": dict(zip(names, nouns)),
        "outputs": dict(zip(names, outputs)),
    }


# ════════════════════════════════════════════════════════════════════════════
# The stylised job-applications server
# ════════════════════════════════════════════════════════════════════════════
#
# Ten tools, ten lexically distinct verbs — score, extract, rewrite, produce,
# write, translate, return, set, apply, build. That list is the reason the
# verb-and-output split was abandoned: it scored all ten at 100% distinct,
# including the three the author had already flagged.
#
# The four descriptions with published shares are verbatim from the write-up.
STYLISED = [
    tool("score_resume", "Score a resume for ATS compatibility.",
         ["resume_text", "job_description"]),
    tool("analyze_job_description", "Extract what a job posting actually screens on.",
         ["job_description"]),
    tool("optimize_resume", "Rewrite a resume so it passes ATS screening.",
         ["resume_text", "job_description"]),
    tool("generate_cover_letter", "Produce a cover letter for this job from the resume.",
         ["resume_text", "job_description"]),
    tool("write_outreach_note", "Write a short note to the hiring manager.",
         ["company", "role"]),
    tool("translate_resume", "Translate a resume into another language.",
         ["resume_text", "language"]),
    tool("search_jobs", "Return the user's job matches.", ["query"]),
    tool("update_job_preferences", "Set the roles and locations it hunts for.",
         ["roles", "locations"]),
    tool("apply_to_job", "Apply to a job posting on the user's behalf.",
         ["job_id", "resume_text"]),
    tool("build_profile", "Build a candidate profile from scratch.", ["full_name"]),
]

group("A — the pair the stemming correction was reported on")
check("screens and screening are one word",
      A.record("screens") == A.record("screening"), True)
shared_words = (set(A.tokenize(STYLISED[1]["description"]))
                & set(A.tokenize(STYLISED[2]["description"])))
check("and the two tools now share it", A.record("screening") in shared_words, True)
note("before stemming these two shared no content word at all")

group("B — input overlap sees what distinctive share cannot")
sty = measure(STYLISED)

check("score_resume / optimize_resume shares its object",
      sty["pairs"][("optimize_resume", "score_resume")]["shares_inputs"], True)
# ── The pair input overlap does NOT catch, asserted as such ─────────────────
#
# The specification for this signal expected this pair to flag. It does not, and
# it should not be made to: `analyze_job_description` takes a job description and
# nothing else, while `score_resume` takes a resume as well, so the two do not
# act on the same set of things. What they share is a *request* — "check my
# resume for this job" is a fair sentence for both — and a request is not in the
# schema.
#
# This is the same pair the contrastive-prompt work names as the ceiling every
# static signal here runs into. Pinning it as a non-flag is the honest version of
# that claim: the gap is real, it is measured, and no threshold closes it.
pair = sty["pairs"][("analyze_job_description", "score_resume")]
check("score_resume / analyze_job_description does NOT share its object",
      pair["shares_inputs"], False)
note(f"overlap {pair['overlap']} — they share a request, not an object")
note("no static signal in this file reaches this pair; see the contrastive harness")
check("update_job_preferences / translate_resume does not",
      sty["pairs"][("translate_resume", "update_job_preferences")]["shares_inputs"], False)

# The reason the signal exists at all: the metric already in place calls the
# ambiguous cluster healthy. If this ever fails, input overlap has stopped
# measuring anything distinctive share does not already measure.
for name in ("score_resume", "analyze_job_description", "optimize_resume"):
    check(f"{name} still clears the corpus median on distinctive share",
          sty["distinctive"][name] >= A.CORPUS_MEDIAN_DISTINCTIVE, True)

group("C — a well-designed pair that shares an input is not flagged")

# The counter-example, from a six-tool ticket server. Identical input shapes, and
# correctly two tools: one returns flat details for many tickets, the other a
# bounded evidence bundle with comments and source anchors.
TICKETS = [
    tool("get_tickets", "Fetch tickets by id. Returns status, assignee, priority and title.",
         ["ticket_ids"], output=["status", "assignee", "priority", "title"]),
    tool("build_ticket_evidence",
         "Assemble an evidence bundle for tickets. Returns comments, attachments and source anchors.",
         ["ticket_ids"], output=["comments", "attachments", "anchors"]),
    # And the failure the gate must still catch: same object, and neither tool
    # says what it gives back.
    tool("list_tickets", "List the tickets in a project.", ["project_id", "ticket_ids"]),
    tool("show_tickets", "Show the tickets in a project.", ["project_id", "ticket_ids"]),
]
tick = measure(TICKETS)

check("the well-designed pair shares its input",
      tick["pairs"][("build_ticket_evidence", "get_tickets")]["shares_inputs"], True)
check("...and is NOT flagged, because the outputs differ",
      tick["pairs"][("build_ticket_evidence", "get_tickets")]["flagged"], False)
check("...it is reported as a near miss instead",
      tick["pairs"][("build_ticket_evidence", "get_tickets")]["near_miss"], True)
check("the silent pair shares its input",
      tick["pairs"][("list_tickets", "show_tickets")]["shares_inputs"], True)
check("...and IS flagged, because neither states an output",
      tick["pairs"][("list_tickets", "show_tickets")]["flagged"], True)

# Silence is not a distinction. The whole gate turns on this line.
check("an unstated output is not distinguishable from anything",
      A.outputs_distinguishable(set(), {"comments", "anchors"}), False)

group("D — the domain noun and the cluster noun are different things")
tightest = {A.surfaces([w])[0]: sorted(m) for w, m in sty["clusters"] if len(m) == 2}
check("the tightest cluster is ats",
      tightest.get("ats"), ["optimize_resume", "score_resume"])
note("the pair the author flagged first, and the pair no ratio in this file names")

# The trap: stripping frequent nouns globally would delete exactly this.
check("resume is a cluster, not stripped as the domain",
      A.record("resume") not in sty["domain"], True)
# The widest cluster the server has, which is what a focused server looks like
# from the inside: a third of its tools act on the same object.
widest = max((len(m), A.surfaces([w])[0]) for w, m in sty["clusters"])
check("...and it is the widest cluster on the server", widest, (4, "resume"))

# Dial-free by construction: the domain is what *every* tool says, so a
# ten-tool server with no universal word simply has no domain to strip.
check("no word is on all ten tools, so nothing is stripped", sty["domain"], set())

# ════════════════════════════════════════════════════════════════════════════
# The same server, as the registry actually holds it
# ════════════════════════════════════════════════════════════════════════════
group("the real server, not the paraphrase")

REAL_PATH = os.path.join(HERE, "..", "data", "fixture_servers.json")
if not os.path.exists(REAL_PATH):
    note("data/fixture_servers.json missing — run scripts/06_extract_fixture.py")
else:
    real = measure(json.load(open(REAL_PATH))["firstexhotic/ai-applyd"])

    # ── input overlap is length-sensitive, and this is where that shows ──────
    #
    # Jaccard divides by the nouns *either* tool uses. The stylised descriptions
    # are one line each and share most of what little they have; the real ones
    # run to three sentences of caveats about credits and authentication, and
    # the shared object is diluted by everything else the author wrote.
    #
    # `score_resume` and `optimize_resume` share seven nouns — ats, compatible,
    # description, job, required, resume, text — and still score 0.269, under
    # the 0.5 bar. So on the very server that motivated this signal, with its
    # real text, input overlap flags nothing.
    #
    # Recorded rather than tuned away. Moving the threshold to catch this pair
    # would move it for 564,810 pairs, and the corpus figure is published.
    pair = real["pairs"][("optimize_resume", "score_resume")]
    check("the real descriptions dilute the overlap below the bar",
          pair["shares_inputs"], False)
    note(f"overlap {pair['overlap']} against a 0.5 bar, on 7 shared nouns")
    note("the paraphrase scores above it; the real text does not. Length, not meaning.")

    # ── and the output gate would have cleared it anyway ─────────────────────
    #
    # The stronger result. These two tools do say what they return — "keyword
    # match score, missing keywords, section scores" against "the optimized
    # content plus a summary of changes" — and those share nothing at all. So
    # even above the input bar, C would not flag this pair, and C would be
    # right: the ambiguity its author reports is between a *user's sentence*
    # and two tools, not between two schemas. That is the ceiling V exists to
    # get past, stated as a passing test.
    check("...and their stated outputs are completely disjoint",
          pair["outputs_distinguishable"], True)
    check("so the output gate would clear the pair regardless",
          pair["flagged"], False)

    # D still finds something on the real server, because D does not depend on
    # a threshold.
    check("one word is on all ten real tools", len(real["domain"]), 1)
    note(f"the domain noun is {sorted(A.surfaces(real['domain']))}")

print(f"\n{'all %d checks passed' % checks if not failures else '%d of %d checks FAILED' % (len(failures), checks)}")
sys.exit(1 if failures else 0)
