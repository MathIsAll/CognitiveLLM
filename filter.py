import json
from tqdm import tqdm

with open("cognition_labeled_instructions/understand_cognition_instructions.json", "r") as f:
    instructions = json.load(f)

new_instructions = []
for instruction in tqdm(instructions):
    instruction["cog_domain"] = "understand"
    new_instructions.append(instruction)

with open("cognition_labeled_instructions/new_understand_cognition_instructions.json", "w", encoding="utf-8") as f:
    json.dump(new_instructions, f, ensure_ascii=False, indent=4)
