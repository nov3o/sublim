import json
from collections import Counter
from pathlib import Path
from loguru import logger

# Find all animal_evaluation*.jsonl files in data subdirectories
data_dir = Path("./data")
eval_files = list(data_dir.glob("**/animal_evaluation*.jsonl"))

logger.info(f"Found {len(eval_files)} evaluation files")

results = {}

for file_path in eval_files:
    # Extract setting and evaluation name from path
    # e.g., ./data/owl_demo/animal_evaluation_results.jsonl
    setting = file_path.parent.name.replace("_demo", "")
    eval_name = file_path.stem.replace("_results", "")

    logger.info(f"Processing {setting}/{eval_name}: {file_path}")

    # Extract completions
    completions = []
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            for response in data['responses']:
                completions.append(response['response']['completion'])

    # Count unique completions and sort by count descending
    counts = Counter(completions)
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    # Limit to top 20 and truncate keys to 40 chars
    top_20 = {}
    for i, (key, value) in enumerate(sorted_counts.items()):
        if i >= 20:
            break
        truncated_key = key[:40] + "..." if len(key) > 40 else key
        top_20[truncated_key] = value

    # Store in nested structure
    if setting not in results:
        results[setting] = {}
    results[setting][eval_name] = top_20

    logger.info(f"{setting}/{eval_name}: {sum(counts.values())} total, {len(counts)} unique, showing top 20")

# Save combined results
output_file = Path("./data/combined_animal_results.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

logger.success(f"Combined results saved to: {output_file}")
