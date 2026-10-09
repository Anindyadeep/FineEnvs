"""Start vLLM in the board's non-thinking configuration, for training or for evaluation."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.sync_grpo import MODEL_REVISIONS, tokenizer_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B", help="Base model; also the served name")
    parser.add_argument("--checkpoint", help="Serve these (merged) weights under the base model's name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--eval", action="store_true", help="Two data-parallel replicas, prefix caching on")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    template = output / "chat_template.jinja"
    template.write_text(tokenizer_for(args.model).chat_template)
    command = [
        "vllm", "serve", args.checkpoint or args.model,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--served-model-name", args.model,
        "--tensor-parallel-size", "1",
        "--data-parallel-size", "2" if args.eval else "1",
        "--dtype", "bfloat16",
        "--max-model-len", "65536",
        "--gpu-memory-utilization", "0.85",
        # Evaluation parses tool calls server-side, as the board's providers do.
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
        "--chat-template", str(template),
        "--default-chat-template-kwargs", '{"enable_thinking":false,"preserve_thinking":true}',
        "--generation-config", "vllm",
        # Qwen3.5-architecture checkpoints are multimodal; this agent sends text only.
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
        "--gdn-prefill-backend", "triton",
    ]  # fmt: skip
    if args.checkpoint:
        command += ["--safetensors-load-strategy", "eager"]  # read sequentially, not memory-mapped
    else:
        command += ["--revision", MODEL_REVISIONS[args.model]]
    if not args.eval:
        # Training: the trainer pushes weights over NCCL after each step, and the token ids and
        # logprobs come back with every completion. Stale prefix-cache entries would outlive a
        # weight update, so the cache stays off, as in 05-multi-harness-rl.
        command += [
            "--weight-transfer-config", '{"backend":"nccl"}',
            "--logprobs-mode", "processed_logprobs",
            "--return-tokens-as-token-ids",
            "--enforce-eager",
            "--no-enable-prefix-caching",
        ]  # fmt: skip
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
