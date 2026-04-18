"""O-PEaR guide model client.

Synchronous client that calls GPT-5.4-nano to generate contrastive (compliant
vs. violating) response pairs for selected rollouts. Uses ThreadPoolExecutor
for parallelism to avoid asyncio/uvloop conflicts inside Ray workers.
"""

import logging
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI

from verl071.opear.prompts import build_guide_prompt, parse_guide_response

logger = logging.getLogger(__name__)


class OPEaRGuide:
    """Synchronous guide model client for O-PEaR contrastive pair generation."""

    def __init__(
        self,
        model: str = "gpt-5.4-nano",
        max_completion_tokens: int = 4096,
        temperature: float = 0.7,
        max_concurrent: int = 32,
    ):
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.max_concurrent = max_concurrent

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("OPENAI_API_KEY")
            except ImportError:
                pass
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")

        self._client = OpenAI(api_key=api_key)

    def _call_guide(self, messages: list[dict], expected_turns: int) -> Optional[list[dict]]:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_completion_tokens,
                    temperature=self.temperature,
                )
                response_text = response.choices[0].message.content
                return parse_guide_response(response_text, expected_turns)
            except Exception as exc:
                logger.warning("Guide call attempt %d/%d failed: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(0.5 * attempt)
        return None

    def generate_pair(self, turns: list[dict], task_description: str, facts: str) -> Optional[dict]:
        expected_turns = sum(1 for t in turns if t["role"] == "assistant")
        if expected_turns == 0:
            logger.warning("No assistant turns found in trajectory; skipping.")
            return None

        compliant_msgs = build_guide_prompt(turns, task_description, "compliant", facts)
        violating_msgs = build_guide_prompt(turns, task_description, "violating", facts)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_c = pool.submit(self._call_guide, compliant_msgs, expected_turns)
            f_v = pool.submit(self._call_guide, violating_msgs, expected_turns)
            compliant_result = f_c.result()
            violating_result = f_v.result()

        if compliant_result is None or violating_result is None:
            logger.warning(
                "Failed to generate pair: compliant=%s, violating=%s",
                compliant_result is not None, violating_result is not None,
            )
            return None

        return {"compliant": compliant_result, "violating": violating_result}

    def generate_contrastive_batch(self, trajectories: list[dict]) -> list[Optional[dict]]:
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {
                pool.submit(
                    self.generate_pair,
                    turns=traj["turns"],
                    task_description=traj["task_description"],
                    facts=traj["facts"],
                ): i
                for i, traj in enumerate(trajectories)
            }
            results: list[Optional[dict]] = [None] * len(trajectories)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.warning("generate_pair failed for trajectory %d: %s", idx, exc)

        elapsed = time.time() - t0
        success_count = sum(1 for r in results if r is not None)
        print(f"[O-PEaR] guide model: {success_count}/{len(trajectories)} pairs in {elapsed:.1f}s")
        return results
