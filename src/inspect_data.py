import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

source = Path("data/longmemeval_s_cleaned.json")
with source.open(encoding="utf-8") as stream:
    rows = json.load(stream)
assert rows, "Dataset is empty"
ids = [r["question_id"] for r in rows]
assert len(ids) == len(set(ids)), "Duplicate question IDs"
groups = defaultdict(list)
for row in rows:
    assert len(row["haystack_sessions"]) == len(row["haystack_dates"])
    assert len(row["haystack_sessions"]) == len(row["haystack_session_ids"])
    category = ("abstention" if row["question_id"].endswith("_abs")
                else row["question_type"])
    groups[category].append(row)
lengths = [len(r["haystack_sessions"]) for r in rows]
summary = {
    "questions": len(rows),
    "counts_by_reporting_group": {k: len(v) for k, v in groups.items()},
    "sessions_min": min(lengths), "sessions_mean": mean(lengths),
    "sessions_max": max(lengths),
}
rng = random.Random(42)
pilot_ids = []
for category in sorted(groups):
    group = sorted(groups[category], key=lambda r: r["question_id"])
    pilot_ids.extend(r["question_id"] for r in
                      rng.sample(group, min(2, len(group))))
Path("results").mkdir(exist_ok=True)
Path("results/dataset_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
Path("results/pilot_ids.json").write_text(
    json.dumps(pilot_ids, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("Pilot questions:", len(pilot_ids))
print("Example question:", rows[0]["question"])
