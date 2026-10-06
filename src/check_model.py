import json
import time
import urllib.request
from pathlib import Path

payload = {
    "model": "llama3.1:8b",
    "prompt": "Reply with one short sentence about memory.",
    "stream": False,
    "options": {
        "temperature": 0, "seed": 42,
        "num_ctx": 4096, "num_predict": 128,
    },
}
request = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
started = time.perf_counter()
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.load(response)
elapsed = time.perf_counter() - started
record = {"request": payload, "response": result,
          "elapsed_seconds": elapsed}
Path("results").mkdir(exist_ok=True)
Path("results/model_check.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8"
)
print(result["response"])
print("Input tokens:", result.get("prompt_eval_count"))
print("Output tokens:", result.get("eval_count"))
print("Elapsed seconds:", round(elapsed, 3))
