"""
Does the registry hand back the schema the server actually serves?

Everything in this study is computed on schemas Smithery stored, not on
schemas we watched a server emit. That is only sound if the two are the same
object. A first look said they might not be: across 1,377 tools in the smoke
sample, **not one** carried a top-level `required` array. Real MCP servers mark
parameters required all the time, so either the corpus is extraordinary or
something in the pipeline drops the field.

Which of those it is decides whether the post exists. If the registry
normalises schemas, then a study of registry schemas measures the registry.

So: boot the official reference servers locally, take their real `tools/list`
over stdio, and diff field-by-field against what the registry reports for the
same packages. No network is needed by any of these, and none takes a
credential — they are the only servers where ground truth is free.

Run:  python3 scripts/02_validate_ground_truth.py
"""

import json
import os
import subprocess
import sys
import time

OUT = os.path.join(os.path.dirname(__file__), "..", "data")

# Official reference servers that boot with no credentials and no network.
REFERENCE = [
    ("sequential-thinking", ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"]),
    ("memory", ["npx", "-y", "@modelcontextprotocol/server-memory"]),
    ("everything", ["npx", "-y", "@modelcontextprotocol/server-everything"]),
    ("filesystem", ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
]


def rpc(proc, payload):
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()


def read_reply(proc, want_id, timeout=90):
    """Read stdio until the reply with `want_id` arrives.

    Servers write banners and logs to stdout as well as protocol frames, which
    is itself a bug but not this study's — so non-JSON lines are skipped
    rather than treated as a failure.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("id") == want_id:
            return msg
    return None


def list_tools(cmd):
    """Boot one server over stdio and return its real `tools/list`."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        rpc(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-schema-study", "version": "1.0"},
            },
        })
        if read_reply(proc, 1) is None:
            return None

        rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})

        reply = read_reply(proc, 2)
        if not reply or "result" not in reply:
            return None
        return reply["result"].get("tools", [])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def summarise(tools):
    """The three facts the study depends on, per tool."""
    out = {}
    for t in tools:
        schema = t.get("inputSchema") or {}
        props = schema.get("properties") or {}
        out[t["name"]] = {
            "n_params": len(props),
            "has_required": bool(schema.get("required")),
            "n_required": len(schema.get("required") or []),
            "n_described": sum(1 for p in props.values()
                               if isinstance(p, dict) and p.get("description")),
            "n_enum": sum(1 for p in props.values()
                          if isinstance(p, dict) and p.get("enum")),
            "has_description": bool(t.get("description")),
        }
    return out


def main():
    truth = {}
    for name, cmd in REFERENCE:
        print(f"booting {name} …", flush=True)
        try:
            tools = list_tools(cmd)
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        if tools is None:
            print("  no tools/list reply")
            continue
        truth[name] = summarise(tools)
        withreq = sum(1 for v in truth[name].values() if v["has_required"])
        print(f"  {len(tools)} tools, {withreq} with a top-level `required`")

    with open(os.path.join(OUT, "ground_truth.json"), "w") as f:
        json.dump(truth, f, indent=1)

    total = sum(len(v) for v in truth.values())
    withreq = sum(1 for v in truth.values() for t in v.values() if t["has_required"])
    print(f"\nground truth: {total} tools across {len(truth)} servers")
    print(f"  carry a top-level `required` : {withreq}")

    if total and withreq == 0:
        print("\n  → real servers also omit `required`. The registry is not the cause.")
    elif withreq:
        print(f"\n  → real servers DO emit `required` ({withreq}/{total}), and the "
              f"registry reported it for none.\n"
              f"    The registry normalises schemas. `required` is unusable, and every\n"
              f"    other field needs checking before it is trusted.")


if __name__ == "__main__":
    sys.exit(main())
