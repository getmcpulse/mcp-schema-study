"""
Collect MCP server tool schemas from the Smithery registry.

Why the registry rather than booting containers
-----------------------------------------------
The obvious method is to boot each server in an isolated container and call
`tools/list` over stdio. It is the most faithful method and it is also the one
that silently biases the sample: a large share of public MCP servers refuse to
start without real credentials, so the servers you end up measuring are the
ones that need no API key — which is not a random subset of anything.

Smithery already stores the `tools/list` response for every server it has
scanned, `inputSchema` included. That is byte-for-byte the JSON a model is
handed, which is the only object this study looks at, and it is reproducible
by anyone with curl.

What it costs us, stated rather than hidden: we inherit Smithery's coverage
and its scan freshness, and we learn nothing about whether a server boots.

Two populations, because the registry will not enumerate
--------------------------------------------------------
`/servers` caps pagination at 5 pages of 100. Page 1 opens in the tens of
thousands of installs and page 5 ends near 1,700, so those 500 are, near
enough, **the most-installed servers on the registry**.

That cap is why there is a second population. `?q=` returns up to ~200 matches
per term, so a union of a few hundred terms reaches deep into the long tail.
It is not a random sample either — it is whatever the search ranker surfaces —
but it is a different bias from popularity, which is the point of having both.

  POPULAR   the ~500 most-installed. The servers a model is most likely to
            actually meet, and the conservative read: a widely-installed
            server has had more eyes on its schema.
  LONGTAIL  the search union. Everything else the registry will show us.

Every figure in the post is reported for both. Where they disagree, that
disagreement is the finding.

Output
------
data/frame.jsonl   one line per server in the sampling frame, with `pool`
data/tools.jsonl   one line per server we got a detail response for
"""

import itertools
import json
import os
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE = "https://registry.smithery.ai"
OUT = os.path.join(os.path.dirname(__file__), "..", "data")

# Deliberately gentle. This is somebody else's free API and the whole study is
# one afternoon of reads — there is no version of this worth rate-limiting
# them over.
CONCURRENCY = 8
PAGE_SIZE = 100
RETRIES = 3
# The registry stops returning rows past this; discovered, not documented.
MAX_LIST_PAGES = 5

lock = threading.Lock()


def search_terms():
    """A few hundred queries chosen to overlap heavily and miss little.

    Bigrams over consonant+vowel catch most English-ish names; the domain
    words catch the naming conventions MCP servers actually use. Overlap is
    free — everything is deduplicated by `qualifiedName`.
    """
    bigrams = ["".join(p) for p in itertools.product(string.ascii_lowercase, "aeiou")]
    domains = [
        "github", "gitlab", "slack", "discord", "notion", "linear", "jira",
        "aws", "gcp", "azure", "cloudflare", "vercel", "docker", "kubernetes",
        "sql", "postgres", "mysql", "sqlite", "mongo", "redis", "supabase",
        "file", "fs", "search", "browser", "web", "scrape", "crawl", "fetch",
        "api", "http", "rest", "graphql", "weather", "maps", "calendar",
        "mail", "gmail", "email", "sheets", "docs", "drive", "s3", "storage",
        "crypto", "stripe", "payment", "shopify", "hubspot", "salesforce",
        "openai", "anthropic", "llm", "agent", "memory", "vector", "rag",
        "image", "video", "audio", "pdf", "excel", "csv", "json",
        "test", "deploy", "ci", "log", "monitor", "metric", "trace",
        "twitter", "reddit", "youtube", "news", "rss", "translate",
        "time", "date", "shell", "terminal", "code", "git", "npm", "python",
    ]
    return list(dict.fromkeys(bigrams + domains + list(string.ascii_lowercase)))


def get(client, url, params=None):
    """One GET with retries. Returns None rather than raising: a single dead
    server must not end a run that has already cost thousands of requests."""
    for attempt in range(RETRIES):
        try:
            r = client.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code >= 500:
                time.sleep(1 + attempt)
                continue
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            time.sleep(1 + attempt)
    return None


def collect_popular(client):
    """Pages 1..5 of the default listing, which is roughly install order."""
    rows = []
    for p in range(1, MAX_LIST_PAGES + 1):
        d = get(client, f"{BASE}/servers", {"page": p, "pageSize": PAGE_SIZE})
        batch = d.get("servers", []) if d else []
        if not batch:
            break
        rows.extend(batch)
    print(f"POPULAR  : {len(rows)}", flush=True)
    return rows


def collect_longtail(client, exclude):
    """Union over search terms, minus anything already in POPULAR."""
    terms = search_terms()
    found = {}

    def one(term):
        out = []
        for p in (1, 2):
            d = get(client, f"{BASE}/servers", {"q": term, "page": p, "pageSize": PAGE_SIZE})
            batch = d.get("servers", []) if d else []
            if not batch:
                break
            out.extend(batch)
        return out

    done = [0]
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        for rows in pool.map(one, terms):
            with lock:
                done[0] += 1
                for s in rows:
                    name = s.get("qualifiedName")
                    if name and name not in exclude:
                        found[name] = s
                if done[0] % 50 == 0:
                    print(f"  searched {done[0]}/{len(terms)} → {len(found)} new", flush=True)

    print(f"LONGTAIL : {len(found)} (from {len(terms)} terms)", flush=True)
    return list(found.values())


def collect_details(client, names):
    """Per-server detail, which is where `tools` lives."""
    done = [0]
    out = []

    def one(name):
        d = get(client, f"{BASE}/servers/{name}")
        with lock:
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  detailed {done[0]}/{len(names)}", flush=True)
        if not d:
            return {"qualifiedName": name, "detail_failed": True, "tools": None}
        return {
            "qualifiedName": name,
            "displayName": d.get("displayName"),
            "description": d.get("description"),
            "remote": d.get("remote"),
            "security": d.get("security"),
            "connections": d.get("connections"),
            # null when Smithery has never successfully scanned the server.
            # That is a different thing from a server with zero tools, and the
            # analysis has to keep them apart.
            "tools": d.get("tools"),
        }

    with ThreadPoolExecutor(CONCURRENCY) as pool:
        for r in pool.map(one, names):
            if r:
                out.append(r)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    headers = {"user-agent": "mcp-schema-study/1.0 (research; contact via repo)"}

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        popular = collect_popular(client)
        pop_names = {s["qualifiedName"] for s in popular if s.get("qualifiedName")}
        longtail = collect_longtail(client, pop_names)

        frame = []
        for s in popular:
            s["pool"] = "popular"
            frame.append(s)
        for s in longtail:
            s["pool"] = "longtail"
            frame.append(s)

        with open(os.path.join(OUT, "frame.jsonl"), "w") as f:
            for s in frame:
                f.write(json.dumps(s) + "\n")

        names = [s["qualifiedName"] for s in frame if s.get("qualifiedName")]
        print(f"\nfetching detail for {len(names)} servers", flush=True)
        details = collect_details(client, names)

        pool_of = {s["qualifiedName"]: s["pool"] for s in frame if s.get("qualifiedName")}
        for d in details:
            d["pool"] = pool_of.get(d["qualifiedName"])

        with open(os.path.join(OUT, "tools.jsonl"), "w") as f:
            for d in details:
                f.write(json.dumps(d) + "\n")

    withtools = sum(1 for d in details if d.get("tools"))
    failed = sum(1 for d in details if d.get("detail_failed"))
    print(f"\nframe          : {len(frame)}")
    print(f"detail fetched : {len(details) - failed}")
    print(f"detail failed  : {failed}")
    print(f"with schemas   : {withtools}")
    print(f"null tools     : {len(details) - withtools - failed}")


if __name__ == "__main__":
    main()
