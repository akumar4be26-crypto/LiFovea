"""Inline the demo payload into the dashboard template.

Two outputs from one template:

* ``--mode artifact`` - page content only (no doctype/head/body), which is
  what the Artifact publisher wraps in its own skeleton.
* ``--mode standalone`` - a complete HTML document that opens from disk.
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "web", "dashboard.html")
PAYLOAD = os.path.join(ROOT, "web", "demo_data.json")

SKELETON = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>:root{{color-scheme:light dark}}body{{margin:0}}img{{max-width:100%}}
[hidden]{{display:none!important}}</style>
</head><body>
{body}
</body></html>"""


def build(mode: str, out: str, payload_path: str = PAYLOAD) -> str:
    with open(TEMPLATE) as fh:
        body = fh.read()
    with open(payload_path) as fh:
        payload = json.load(fh)
    body = body.replace("__AVR_DATA__", json.dumps(payload, separators=(",", ":")))
    html = body if mode == "artifact" else SKELETON.format(body=body)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["artifact", "standalone"], default="artifact")
    ap.add_argument("--out", default="build/dashboard.html")
    ap.add_argument("--payload", default=PAYLOAD)
    args = ap.parse_args(argv)
    path = build(args.mode, args.out, args.payload)
    print(f"wrote {path}  ({os.path.getsize(path) / 1e6:.2f} MB, mode={args.mode})")


if __name__ == "__main__":
    main()
