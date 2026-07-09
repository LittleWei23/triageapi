"""
Step 1.1 — Generate a synthetic support-ticket dataset with Cohere Command.

No .env file needed. The script checks for a COHERE_API_KEY environment
variable, and if it isn't set, just asks you to paste your key when you
run it (input is hidden). This sidesteps all the file-encoding/quoting
issues that come from trying to write a .env file by hand.

Usage:
    pip install cohere
    python generate_tickets.py
    python generate_tickets.py --per-batch 5 --batches-per-category 1   # small test run first

Optional (skip the prompt on every run): set the key for your current
terminal session only (nothing written to disk):
    PowerShell:  $env:COHERE_API_KEY = "your-key-here"
    macOS/Linux: export COHERE_API_KEY="your-key-here"

Outputs:
    api/data/tickets.jsonl      corpus for embedding/ingestion
    api/data/golden_set.jsonl   held-out queries for the Step 1.4 eval
"""

import argparse
import getpass
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import cohere

# ---------------------------------------------------------------- config

CATEGORIES = {
    "billing":          "invoices, refunds, unexpected charges, plan changes, payment failures",
    "authentication":   "login failures, SSO problems, password resets, 2FA issues, locked accounts",
    "api_errors":       "4xx/5xx responses, rate limits, timeouts, malformed payloads, webhook delivery failures",
    "integration":      "connecting third-party systems, data sync problems, field mapping confusion, SDK setup",
    "feature_request":  "asking for new capabilities, endpoints, configuration options, or UI improvements",
    "performance":      "slow responses, latency spikes, degraded throughput, timeouts under load",
    "data_quality":     "wrong or missing records, duplicates, encoding issues, bad exports",
    "account_admin":    "user management, permissions, seat limits, org settings, offboarding",
}

PRIORITY_WEIGHTS = [("low", 0.3), ("normal", 0.4), ("high", 0.2), ("urgent", 0.1)]

PROMPT_TEMPLATE = """You are generating realistic B2B SaaS customer-support tickets
for the category "{category}" ({description}).

Generate {n} tickets. Vary tone (frustrated, polite, terse, verbose), customer
technical skill, and length (1-6 sentences in the body). Include realistic but
FAKE details: error codes, endpoint paths, company names, timestamps. Never use
real personal data.

Respond with ONLY a JSON array, no markdown fences, no commentary:
[{{"subject": "...", "body": "..."}}, ...]"""

# ---------------------------------------------------------------- helpers

def get_api_key() -> str:
    """Check env var first; otherwise prompt (hidden input, nothing written to disk)."""
    key = os.environ.get("COHERE_API_KEY")
    if key:
        print("Using COHERE_API_KEY from environment.")
        return key.strip()

    print("No COHERE_API_KEY environment variable found.")
    key = getpass.getpass("Paste your Cohere API key (input hidden, Enter to submit): ").strip()
    if not key:
        raise SystemExit("No key entered. Get one free at https://dashboard.cohere.com/api-keys")
    return key


def extract_json_array(text: str):
    """Command usually complies, but strip fences / surrounding prose defensively."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in model output")
    return json.loads(text[start : end + 1])


def weighted_priority() -> str:
    r, cum = random.random(), 0.0
    for value, weight in PRIORITY_WEIGHTS:
        cum += weight
        if r <= cum:
            return value
    return "normal"


def dedupe_key(subject: str, body: str) -> str:
    normalized = re.sub(r"\W+", "", (subject + body).lower())
    return hashlib.md5(normalized.encode()).hexdigest()

# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-batch", type=int, default=10,
                        help="tickets requested per API call (10 keeps outputs parseable)")
    parser.add_argument("--batches-per-category", type=int, default=5,
                        help="5 batches x 10 tickets x 8 categories = ~400 tickets")
    parser.add_argument("--eval-holdout", type=int, default=8,
                        help="tickets per category reserved for the golden eval set")
    parser.add_argument("--out-dir", default="api/data")
    args = parser.parse_args()

    random.seed(42)  # reproducible priorities + holdout split
    api_key = get_api_key()
    co = cohere.ClientV2(api_key)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    tickets: list[dict] = []

    for category, description in CATEGORIES.items():
        for batch in range(args.batches_per_category):
            prompt = PROMPT_TEMPLATE.format(
                category=category, description=description, n=args.per_batch
            )
            try:
                response = co.chat(
                    model="command-r-08-2024",   # check dashboard for current model names
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,             # high temp = more variety
                )
                raw = response.message.content[0].text
                items = extract_json_array(raw)
            except Exception as exc:  # noqa: BLE001 - log and continue on any batch failure
                print(f"  [warn] {category} batch {batch + 1} failed: {exc}")
                time.sleep(5)
                continue

            added = 0
            for item in items:
                subject = str(item.get("subject", "")).strip()
                body = str(item.get("body", "")).strip()
                if len(subject) < 5 or len(body) < 20:
                    continue                      # reject junk rows
                key = dedupe_key(subject, body)
                if key in seen:
                    continue                      # reject near-duplicates
                seen.add(key)
                tickets.append(
                    {
                        "id": f"TKT-{len(tickets) + 1:04d}",
                        "subject": subject,
                        "body": body,
                        "category": category,
                        "priority": weighted_priority(),
                    }
                )
                added += 1

            print(f"{category}: batch {batch + 1}/{args.batches_per_category} -> +{added} tickets")
            time.sleep(2)  # be gentle with trial-key rate limits

    # ---- split: hold out an eval set BEFORE ingestion so the eval is honest
    random.shuffle(tickets)
    golden, corpus = [], []
    holdout_count = {c: 0 for c in CATEGORIES}
    for t in tickets:
        if holdout_count[t["category"]] < args.eval_holdout:
            holdout_count[t["category"]] += 1
            golden.append(t)
        else:
            corpus.append(t)

    for name, rows in [("tickets.jsonl", corpus), ("golden_set.jsonl", golden)]:
        path = out_dir / name
        # newline="\n" stops Windows from writing \r\n into a file that will be
        # read inside a Linux container later.
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {len(rows):4d} rows -> {path}")

    print("\nCategory distribution (corpus):")
    for category in CATEGORIES:
        count = sum(1 for t in corpus if t["category"] == category)
        print(f"  {category:16s} {count}")


if __name__ == "__main__":
    main()