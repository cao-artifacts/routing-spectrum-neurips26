# E4b: route-patched WikiText-2 perplexity (full test set, window=stride=512)

phi maps from E4a (selected on the published 200-prompt c4 calibration set); identity = identity-phi run (the harness's 30-gate suite verifies bit-exact identity-phi NLL reproduction on GQA reference models; gates were not re-run on these two checkpoints).

| Model | R | greedy | eigen | similarity | random | identity |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-8b | 4 | 303.5 | 547.1 | 438.1 | 21291.1 | 12.87 |
| qwen3-8b | 8 | 78.7 | 76.7 | 95.5 | 541.9 | 12.87 |
| qwen3-8b | 16 | 20.7 | 26.6 | 26.1 | 635.6 | 12.87 |
| llama3-8b | 4 | 61125.6 | 67925.9 | 35260.6 | 37280.3 | 7.99 |
| llama3-8b | 8 | 1249.9 | 20561.5 | 51652.7 | 67530.9 | 7.99 |
| llama3-8b | 16 | 48.8 | 34794.0 | 353.5 | 529.7 | 7.99 |
