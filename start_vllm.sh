#!/usr/bin/env bash
# start_vllm.sh — launch vLLM with the correct tool-call parser for ATB.
#
# Usage:
#   ./start_vllm.sh <model-name> [extra vllm args...]
#
# Examples:
#   ./start_vllm.sh Qwen/Qwen2.5-14B-Instruct
#   ./start_vllm.sh meta-llama/Llama-3.1-8B-Instruct
#   ./start_vllm.sh hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
#   ./start_vllm.sh Qwen/Qwen2.5-14B-Instruct --max-model-len 4096
#
# The tool-call parser is selected automatically by matching the model
# name. Override by passing `--tool-call-parser <name>` as an extra arg.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model-name> [extra-vllm-args...]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 Qwen/Qwen2.5-14B-Instruct" >&2
    echo "  $0 meta-llama/Llama-3.1-8B-Instruct" >&2
    echo "  $0 hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" >&2
    exit 1
fi

MODEL="$1"
shift  # remaining args pass through to vllm

# --- Pick the right tool-call parser ----------------------------------------
case "$MODEL" in
    *Qwen*|*qwen*)
        TOOL_PARSER="hermes"
        ;;
    *Llama*|*llama*|*Meta-Llama*)
        TOOL_PARSER="llama3_json"
        ;;
    *Mistral*|*mistral*)
        TOOL_PARSER="mistral"
        ;;
    *)
        TOOL_PARSER="hermes"  # reasonable default
        echo "[WARN] Unknown model family for $MODEL; defaulting --tool-call-parser=hermes." >&2
        echo "       Override with: $0 $MODEL --tool-call-parser <name>" >&2
        ;;
esac

# --- Detect quantization from model name ------------------------------------
EXTRA_ARGS=()
if [[ "$MODEL" =~ AWQ ]] || [[ "$MODEL" =~ awq ]]; then
    EXTRA_ARGS+=(--quantization awq)
fi
if [[ "$MODEL" =~ GPTQ ]] || [[ "$MODEL" =~ gptq ]]; then
    EXTRA_ARGS+=(--quantization gptq_marlin)
fi

# --- Detect GPU count for tensor parallel ---------------------------------
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 || echo "1")
GPU_COUNT=${GPU_COUNT:-1}

# --- Launch ----------------------------------------------------------------
echo "[INFO] Launching vLLM"
echo "       Model:        $MODEL"
echo "       Tool parser:  $TOOL_PARSER"
echo "       GPUs:         $GPU_COUNT"
echo "       Extra args:   ${EXTRA_ARGS[*]:-}"
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tool-call-parser "$TOOL_PARSER" \
    --enable-auto-tool-choice \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --tensor-parallel-size "$GPU_COUNT" \
    --enforce-eager \
    "${EXTRA_ARGS[@]}" \
    "$@"
