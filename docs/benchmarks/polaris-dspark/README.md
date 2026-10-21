# Benchmark evidence

`results.json` / `results.csv` contain 18 formal points, 300 requests each.
SGLang uses corrected native FlashInfer draft attention; SFLLM points are reused
from the original validated run. `summaries/` contains exact client summaries.
Re-render: `python plot.py --output /tmp/comparison.png` (matplotlib required).

`commands.json` gives selected server/client commands and environment overrides.
`sglang-selection.json` contains every corrected candidate score before formal
measurement; `sfllm-selection.json` retains the original SFLLM search.
`verification.json` records corrected source/capacity/input/acceptance checks.
`protocol.json` includes sample indices and SHA256; prompt text is private.

`history-triton/` preserves the superseded SGLang curve and native adaptive
cap-accept sweep. Those numbers are not used as corrected FlashInfer results.
The adjacent alignment report explains valid and failed native option checks.
Existing native-cost/command files describe historical calibration, not a
successful corrected compact-adaptive benchmark.

Runtime, hardware, source import, and dependency integrity metadata are retained.
Local absolute paths identify the measured environment and must be adapted on
another host. Full logs and per-request records remain in the experiment folder.
