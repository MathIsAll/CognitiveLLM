from datasets import load_dataset
import json

ds = load_dataset("HuggingFaceH4/aime_2024")

js_ds = []
for item in ds["train"]:
    js_ds += [
        {"problem": item["problem"],
         "answer": item["answer"], },
    ]

with open("data/AIME2024/AIME2024.json", "w", encoding="utf-8") as f:
    json.dump(js_ds, f, ensure_ascii=False, indent=4)
