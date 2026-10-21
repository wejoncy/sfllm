#!/usr/bin/env bash
# DSpark / internal dataset / 1 x H100 NVL. Run one server at a time.
# Set TARGET, DRAFT, SFLLM_REPO and SGLANG_REPO to existing local paths.
# Optional: SFLLM_PYTHON, SGLANG_PYTHON, SFLLM_DEPS, SGLANG_DEPS, GPU, PORT.
# Measured source: SFLLM 17555935c3237ad1cf4afdd756943ae3f503dfa1;
#                  SGLang e262b44eef1f2bf83e34345276e2bb91a086587e (official).
# Measured dependencies (torch / triton / flashinfer / transformers / sglang-kernel):
# SFLLM: 2.13.0+cu130 / 3.7.1 / 0.6.18 / 5.12.1 / 0.4.6.post1
# SGLang: 2.14.1+cu130 / 3.8.0 / 0.7.2 / 5.19.0 / 0.4.10
# Selected server configs at client concurrency C:
# C       1       4       8       12      16      24      32      48      64
# SFLLM   fixed8  fixed8  d8t6    d8t5    d8t6    d8t5    d8t5    fixed4  d8t4.5
# SGLang  fp8-w8  bf16-w8 bf16-w8 bf16-w4 bf16-w4 bf16-w4 bf16-w4 bf16-w4 bf16-w4
# Examples (paths/variables above must be set in each terminal):
#   bash benchmark/reproduce_dspark.sh sfllm d8t5
#   bash benchmark/reproduce_dspark.sh sglang bf16-w4
# In another terminal, DATASET is raw JSONL with one {"prompt":"..."} per line:
#   DATASET=/path/internal-dataset.jsonl OUT=/tmp/sfllm-warmup.jsonl bash benchmark/reproduce_dspark.sh bench 32 32
#   DATASET=/path/internal-dataset.jsonl OUT=/tmp/sglang-warmup.jsonl bash benchmark/reproduce_dspark.sh bench 64 128
# After warming the chosen server, measure its corresponding C values, e.g. SFLLM d8t5:
#   for c in 12 24 32; do
#     DATASET=/path/internal-dataset.jsonl OUT=/tmp/sfllm-d8t5-c${c}.jsonl bash benchmark/reproduce_dspark.sh bench "$c"
#   done
# Each point: 300 requests, max 1000 new tokens, temperature 0, EOS allowed, seed 42.
# The chart used a raw client; bench below reuses the official SGLang client.
# Keep new client results separate; its timings are not the chart's original measurements.
set -euo pipefail
mode=${1:?Usage: reproduce_dspark.sh sfllm CONFIG | sglang CONFIG | bench CONCURRENCY [COUNT]}
: "${TARGET:?Set TARGET to the FP8 target checkpoint}"
: "${SGLANG_REPO:?Set SGLANG_REPO to the official pinned checkout}"
for name in ${!SGLANG_@} ${!SFLLM_@} ${!BENCH_@}; do
  case "$name" in SGLANG_REPO|SFLLM_REPO|SGLANG_PYTHON|SFLLM_PYTHON|SGLANG_DEPS|SFLLM_DEPS) ;; *) unset "$name" ;; esac
done
unset PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/tmp/sfllm-experiments/flashinfer}
port=${PORT:-30000}
[[ $(git -C "$SGLANG_REPO" rev-parse HEAD) == e262b44eef1f2bf83e34345276e2bb91a086587e ]]
[[ -z $(git -C "$SGLANG_REPO" status --porcelain --untracked-files=all) ]]
[[ $(git -C "$SGLANG_REPO" remote get-url origin) == *sgl-project/sglang* ]]
export PYTHONPATH="$SGLANG_REPO/python${SGLANG_DEPS:+:$SGLANG_DEPS}"
py=${SGLANG_PYTHON:-python3}
case "$mode" in
  sfllm)
    : "${SFLLM_REPO:?Set SFLLM_REPO to the measured SFLLM checkout}"
    : "${DRAFT:?Set DRAFT to the DSpark checkpoint}"
    [[ $(git -C "$SFLLM_REPO" rev-parse HEAD) == 17555935c3237ad1cf4afdd756943ae3f503dfa1 ]]
    export PYTHONPATH="$SFLLM_REPO/python:$SFLLM_REPO/sfkernels/python${SFLLM_DEPS:+:$SFLLM_DEPS}"
    export SFLLM_GDN_JOURNAL=1
    case "${2:-d8t5}" in
      fixed4) spec=(--speculative-num-draft-tokens 4) ;;
      fixed8) spec=(--speculative-num-draft-tokens 8) ;;
      d8t4.5|d8t5|d8t6) spec=(--spec-adaptive-verify "${2:-d8t5}") ;;
      *) echo 'SFLLM configs: fixed4 fixed8 d8t4.5 d8t5 d8t6' >&2; exit 2 ;;
    esac
    exec "${SFLLM_PYTHON:-python3}" -B -m sfllm.serving.app \
      --model-path "$TARGET" --speculative-draft-model-path "$DRAFT" \
      --speculative-algorithm dspark --speculative-draft-quantization fp8 \
      --speculative-dspark-topk 16 --dtype bfloat16 --mamba-ssm-dtype float32 \
      --attention-backend fa3 --linear-attn-prefill-backend flashinfer \
      --linear-attn-decode-backend flashinfer --max-running-requests 64 \
      --cuda-graph-max-bs 64 --max-context-length 16384 --mem-fraction 0.9 \
      --tokenizer-worker-num 4 --enable-prefix-cache --port "$port" "${spec[@]}"
    ;;
  sglang)
    : "${DRAFT:?Set DRAFT to the DSpark checkpoint}"
    quant=()
    case "${2:-bf16-w4}" in
      bf16-w4) block=3 ;;
      bf16-w8) block=7 ;;
      fp8-w8) block=7; quant=(--speculative-draft-model-quantization fp8) ;;
      *) echo 'SGLang configs: bf16-w4 bf16-w8 fp8-w8' >&2; exit 2 ;;
    esac
    export SGLANG_DSPARK_USE_GREEDY_DRAFT=1
    export SGLANG_DSPARK_ENABLE_SPS_RECORD=0
    export SGLANG_RAGGED_VERIFY_MODE=static
    exec "$py" -B -m sglang.launch_server \
      --model-path "$TARGET" --speculative-draft-model-path "$DRAFT" \
      --speculative-algorithm DSPARK --speculative-dspark-block-size "$block" \
      --json-model-override-args '{"sample_from_anchor":false}' \
      --dtype bfloat16 --mamba-ssm-dtype float32 --attention-backend fa3 \
      --speculative-draft-attention-backend flashinfer \
      --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer \
      --max-running-requests 80 --cuda-graph-max-bs-decode 64 --context-length 16384 \
      --mem-fraction-static 0.9 --random-seed 42 --port "$port" --skip-server-warmup \
      --max-mamba-cache-size 400 "${quant[@]}"
    ;;
  bench)
    : "${DATASET:?Set DATASET to the internal raw-prompt JSONL}"
    : "${OUT:?Set OUT to a new result JSONL under /tmp}"
    [[ "$OUT" == /tmp/* && ! -e "$OUT" ]]
    mkdir -p "$(dirname "$OUT")"
    exec "$py" -B - --backend sglang --base-url "http://127.0.0.1:$port" \
      --model "$TARGET" --tokenizer "$TARGET" \
      --dataset-name custom --dataset-path "$DATASET" \
      --num-prompts "${3:-300}" --max-concurrency "${2:?Set client concurrency}" \
      --request-rate inf --sharegpt-output-len 1000 --temperature 0 --seed 42 \
      --disable-ignore-eos --warmup-requests 0 --disable-tqdm --flush-cache \
      --extra-request-body '{"sampling_params":{"temperature":0,"max_new_tokens":1000,"stop_token_ids":[248044],"sampling_seed":42,"ignore_eos":false}}' \
      --output-details --output-file "$OUT" <<'PY'
# Only the dataset loader is adapted; scheduling, HTTP/SSE and metrics are official.
import hashlib
import json
import random
from pathlib import Path
from sglang.benchmark import datasets, serving
from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.datasets.custom import CustomDataset

class InternalDataset(CustomDataset):
    def load(self, tokenizer, model_id=None):
        records = [json.loads(s) for s in Path(self.dataset_path).read_text().splitlines() if s.strip()]
        if len(records) == 500:
            records = random.Random(42).sample(records, 300)
        assert len(records) == 300 and 0 < self.num_requests <= 300
        assert all(set(r) == {'prompt'} and isinstance(r['prompt'], str) for r in records)
        data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records).encode()
        assert hashlib.sha256(data).hexdigest() == '8932bd9c7191221b3f953dea3a3addf441f352be9ca60c920710cfd65bec8876', 'Input differs from the measured internal dataset'
        return [DatasetRow(prompt=r['prompt'], prompt_len=len(tokenizer.encode(r['prompt'])), output_len=self.fixed_output_len)
                for r in records[:self.num_requests]]

datasets.DATASET_MAPPING['custom'] = InternalDataset
serving.cli_main()
PY
    ;;
  *) echo 'Modes: sfllm, sglang, bench' >&2; exit 2 ;;
esac
