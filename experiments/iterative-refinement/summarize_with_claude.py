"""Summarize each txt file in ./results using only words from decoder_vocab.json, via the Claude Agent SDK."""

import asyncio
import json
import os
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions
from claude_agent_sdk.types import AssistantMessage, TextBlock, ResultMessage

RESULTS_DIR = Path(__file__).parent / "results"
VOCAB_PATH = RESULTS_DIR / "decoder_vocab.json"


def load_vocab() -> list[str]:
    with open(VOCAB_PATH) as f:
        return json.load(f)


def find_txt_files() -> list[Path]:
    return sorted(RESULTS_DIR.rglob("*.txt"))


async def summarize_file(file_path: Path, vocab_str: str) -> str:
    """Send a txt file to Claude and get a vocab-constrained summary."""
    text = file_path.read_text()

    prompt = (
        f"Summarize the following text in at most 64 words. "
        f"You MUST only use words from this allowed vocabulary list — no other words are permitted:\n\n"
        f"ALLOWED VOCABULARY:\n{vocab_str}\n\n"
        f"TEXT TO SUMMARIZE:\n{text}\n\n"
        f"Write your summary using ONLY words from the allowed vocabulary above. "
        f"Do not use any word not in that list. Output only the summary, nothing else."
    )

    options = ClaudeAgentOptions(
        max_turns=1,
        model="claude-opus-4-6",
    )

    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result_text += block.text

    return result_text.strip()


async def main():
    vocab = load_vocab()
    vocab_str = ", ".join(vocab)
    txt_files = find_txt_files()

    print(f"Found {len(txt_files)} txt files to summarize")
    print(f"Vocabulary size: {len(vocab)} words\n")

    output_dir = RESULTS_DIR / "summaries"
    output_dir.mkdir(exist_ok=True)

    for file_path in txt_files:
        rel_path = file_path.relative_to(RESULTS_DIR)
        print(f"Summarizing: {rel_path} ... ", end="", flush=True)

        try:
            summary = await summarize_file(file_path, vocab_str)

            # Mirror the subdirectory structure in the output
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(summary + "\n")

            print("done")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nSummaries written to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
