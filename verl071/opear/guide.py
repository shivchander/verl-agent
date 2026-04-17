"""O-PEaR guide model client.

Async client that calls GPT-5.4-nano to generate contrastive (compliant vs.
violating) response pairs for selected rollouts. Uses ``max_completion_tokens``
(not ``max_tokens``) as required by GPT-5.4-nano.
"""

import asyncio
import logging
import math
import os
import random
import time
from typing import Optional

from openai import AsyncOpenAI

from verl071.opear.prompts import build_guide_prompt, parse_guide_response

logger = logging.getLogger(__name__)


class OPEaRGuide:
    """Async guide model client for O-PEaR contrastive pair generation."""

    def __init__(
        self,
        model: str = "gpt-5.4-nano",
        beta: float = 0.5,
        max_completion_tokens: int = 4096,
        temperature: float = 0.7,
        max_concurrent: int = 32,
    ):
        self.model = model
        self.beta = beta
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Load API key: try env first, fall back to dotenv
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv

                load_dotenv()
                api_key = os.environ.get("OPENAI_API_KEY")
            except ImportError:
                pass
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment or .env file."
            )

        self._client = AsyncOpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Internal API call with retry
    # ------------------------------------------------------------------

    async def _call_guide(
        self,
        messages: list[dict],
        expected_turns: int,
    ) -> Optional[list[dict]]:
        """Call the guide model and parse the response.

        Retries up to 3 times on any failure (API error or parse error).
        Returns None if all attempts fail.
        """
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=self.max_completion_tokens,
                        temperature=self.temperature,
                    )
                response_text = response.choices[0].message.content
                return parse_guide_response(response_text, expected_turns)
            except Exception as exc:
                logger.warning(
                    "Guide call attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * attempt)
        return None

    # ------------------------------------------------------------------
    # Generate a contrastive pair
    # ------------------------------------------------------------------

    async def generate_pair(
        self,
        turns: list[dict],
        task_description: str,
        facts: str,
    ) -> Optional[dict]:
        """Generate a compliant/violating response pair for one trajectory.

        Args:
            turns: list of {"role": ..., "content": ...} dicts representing
                   the multi-turn trajectory.
            task_description: ALFWorld task description string.
            facts: serialized PDDL facts from the environment.

        Returns:
            {"compliant": [...], "violating": [...]} where each value is a
            list of parsed turn dicts, or None if either call fails.
        """
        # Count the number of assistant turns (those are the ones we rewrite)
        expected_turns = sum(1 for t in turns if t["role"] == "assistant")
        if expected_turns == 0:
            logger.warning("No assistant turns found in trajectory; skipping.")
            return None

        compliant_msgs = build_guide_prompt(
            turns, task_description, "compliant", facts
        )
        violating_msgs = build_guide_prompt(
            turns, task_description, "violating", facts
        )

        compliant_result, violating_result = await asyncio.gather(
            self._call_guide(compliant_msgs, expected_turns),
            self._call_guide(violating_msgs, expected_turns),
        )

        if compliant_result is None or violating_result is None:
            logger.warning(
                "Failed to generate pair: compliant=%s, violating=%s",
                compliant_result is not None,
                violating_result is not None,
            )
            return None

        return {"compliant": compliant_result, "violating": violating_result}

    # ------------------------------------------------------------------
    # Rollout selection
    # ------------------------------------------------------------------

    def select_rollouts(self, traj_uids: list, group_size: int) -> list:
        """Select floor(beta * group_size) unique trajectory UIDs.

        Args:
            traj_uids: list of available trajectory UIDs.
            group_size: the GRPO group size.

        Returns:
            A list of selected UIDs (random sample if more available than
            needed).
        """
        k = math.floor(self.beta * group_size)
        k = min(k, len(traj_uids))
        if k <= 0:
            return []
        return random.sample(traj_uids, k)

    # ------------------------------------------------------------------
    # Synchronous batch wrapper
    # ------------------------------------------------------------------

    def generate_contrastive_batch(
        self,
        trajectories: list[dict],
    ) -> list[Optional[dict]]:
        """Generate contrastive pairs for a batch of trajectories.

        Synchronous wrapper around async generate_pair calls.

        Args:
            trajectories: list of dicts, each with keys "turns",
                "task_description", and "facts".

        Returns:
            List of pair dicts (or None for failures), one per trajectory.
        """

        async def _run_batch() -> list[Optional[dict]]:
            tasks = [
                self.generate_pair(
                    turns=traj["turns"],
                    task_description=traj["task_description"],
                    facts=traj["facts"],
                )
                for traj in trajectories
            ]
            return await asyncio.gather(*tasks)

        t0 = time.time()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply(loop)
            results = loop.run_until_complete(_run_batch())
        else:
            results = asyncio.run(_run_batch())
        elapsed = time.time() - t0

        success_count = sum(1 for r in results if r is not None)
        logger.info(
            "generate_contrastive_batch: %d/%d succeeded in %.2fs",
            success_count,
            len(trajectories),
            elapsed,
        )
        return results
