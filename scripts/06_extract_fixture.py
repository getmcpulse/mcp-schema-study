"""
Pull the named servers out of the corpus and commit them as fixtures.

`05_fixtures.py` needs a handful of real servers to test against, and the corpus
itself is 100 MB and not in the repo. This extracts just those servers so the
fixtures run on a fresh clone with no collection step.

    python3 scripts/01_collect.py          # build the corpus
    python3 scripts/06_extract_fixture.py  # then this

Writes `data/fixture_servers.json`.
"""

import json
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

#: Servers whose authors told us what was wrong with them. That is the only
#: ground truth this study has, so it is worth committing byte for byte.
WANTED = {
    # Ten tools. Its author confirms score_resume, analyze_job_description and
    # optimize_resume are ambiguous in practice, and all three score healthy on
    # distinctive share. The evidence behind the input-overlap signal.
    "firstexhotic/ai-applyd",
}


def main():
    found = {}
    with open(os.path.join(DATA, "tools.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            name = row.get("qualifiedName")
            if name in WANTED and row.get("tools"):
                found[name] = row["tools"]

    missing = WANTED - set(found)
    if missing:
        print(f"not in the corpus: {sorted(missing)}")

    path = os.path.join(DATA, "fixture_servers.json")
    with open(path, "w") as f:
        json.dump(found, f, indent=1)
    for name, tools in found.items():
        print(f"{name}: {len(tools)} tools -> data/fixture_servers.json")


if __name__ == "__main__":
    main()
