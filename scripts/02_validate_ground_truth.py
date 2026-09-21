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


#: The MCP tool hints a client reads when deciding whether to prompt the user.
#:
#: Checked here for the same reason `required` is: the corpus carries the field
#: for **not one** of 82,549 tools, and a study that reported "0% of servers set
#: `destructiveHint`" without establishing why would be publishing a fact about
#: the registry as if it were a fact about MCP authors.
ANNOTATION_HINTS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


def summarise(tools):
    """The facts the study depends on, per tool."""
    out = {}
    for t in tools:
        schema = t.get("inputSchema") or {}
        props = schema.get("properties") or {}
        annotations = t.get("annotations")
        out[t["name"]] = {
            "n_params": len(props),
            "has_required": bool(schema.get("required")),
            "n_required": len(schema.get("required") or []),
            "n_described": sum(1 for p in props.values()
                               if isinstance(p, dict) and p.get("description")),
            "n_enum": sum(1 for p in props.values()
                          if isinstance(p, dict) and p.get("enum")),
            "has_description": bool(t.get("description")),
            # ── the second stripped field, checked the same way as the first ──
            "has_annotations": isinstance(annotations, dict) and bool(annotations),
            "annotation_keys": sorted(annotations) if isinstance(annotations, dict) else [],
            "hints_set": [h for h in ANNOTATION_HINTS
                          if isinstance(annotations, dict) and h in annotations],
            # `outputSchema` survives the registry — 28.5% of sampled tools
            # carry one — so this is a control rather than a suspect: if the
            # reference servers emit it and the registry reports it, the
            # pipeline is not dropping everything it does not understand.
            "has_output_schema": bool(t.get("outputSchema")),
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
        withann = sum(1 for v in truth[name].values() if v["has_annotations"])
        without = sum(1 for v in truth[name].values() if v["has_output_schema"])
        print(f"  {len(tools)} tools, {withreq} with a top-level `required`, "
              f"{withann} with annotations, {without} with an outputSchema")

    with open(os.path.join(OUT, "ground_truth.json"), "w") as f:
        json.dump(truth, f, indent=1)

    total = sum(len(v) for v in truth.values())
    withreq = sum(1 for v in truth.values() for t in v.values() if t["has_required"])
    withann = sum(1 for v in truth.values() for t in v.values() if t["has_annotations"])
    without = sum(1 for v in truth.values() for t in v.values() if t["has_output_schema"])
    hints = sorted({h for v in truth.values() for t in v.values() for h in t["hints_set"]})

    print(f"\nground truth: {total} tools across {len(truth)} servers")
    print(f"  carry a top-level `required` : {withreq}")
    print(f"  carry `annotations`          : {withann}   hints seen: {hints or 'none'}")
    print(f"  carry an `outputSchema`      : {without}")

    if total and withreq == 0:
        print("\n  → real servers also omit `required`. The registry is not the cause.")
    elif withreq:
        print(f"\n  → real servers DO emit `required` ({withreq}/{total}), and the "
              f"registry reported it for none.\n"
              f"    The registry normalises schemas. `required` is unusable, and every\n"
              f"    other field needs checking before it is trusted.")

    # ── annotations, the same question asked again ──────────────────────────
    #
    # The corpus carries `annotations` for none of its 82,549 tools. That is
    # either "MCP authors do not set them" or "the registry drops them", and the
    # two lead to opposite write-ups. Only booting a server settles it.
    if total and withann == 0:
        print("\n  → real servers also omit `annotations`. Nothing can be concluded\n"
              "    about how often authors set them without a different corpus —\n"
              "    these four are reference implementations, not a sample.")
    elif withann:
        print(f"\n  → real servers DO emit `annotations` ({withann}/{total}), and the\n"
              f"    registry reported them for none of 82,549 tools.\n"
              f"    So the annotations check cannot run on this corpus. It runs in the\n"
              f"    checker, where an author pastes their own `tools/list` and the\n"
              f"    field is present.")


if __name__ == "__main__":
    sys.exit(main())
