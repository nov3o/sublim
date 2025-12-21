#!/usr/bin/env python3

import json
import sys
from collections import Counter

if len(sys.argv) != 2:
    print("Usage: python count_unique_responses.py <jsonl_file>")
    sys.exit(1)

file_path = sys.argv[1]

completions = []
with open(file_path) as f:
    for line in f:
        data = json.loads(line)
        for response in data['responses']:
            completions.append(response['response']['completion'])

counts = Counter(completions)
sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

truncated_counts = {}
for key, value in sorted_counts.items():
    truncated_key = key[:40] + "..." if len(key) > 40 else key
    truncated_counts[truncated_key] = value

print(json.dumps(truncated_counts, indent=2, ensure_ascii=False))
