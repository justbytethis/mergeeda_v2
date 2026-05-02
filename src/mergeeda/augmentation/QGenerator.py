"""Question generation module using OpenAI GPT for text chunk question generation."""

import json
import logging
from pathlib import Path

from openai import OpenAI

from mergeeda.utils import MATERIAL_TAG_PATTERN, build_material_content_blocks

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior AMBA protocol expert (AXI, APB, CHI, etc.) with deep experience. Your task is to generate high-quality questions from AMBA specification chunks.

You will receive a text chunk and, optionally, figures and/or tables. Tables are provided as labeled text blocks ([Table: FILENAME]); images are provided inline. Figures/tables are referenced in the text as <material:FILENAME>.

# Goal
Produce questions that an engineer would realistically ask while reading, implementing, or verifying against the spec — questions that build genuine protocol understanding, not surface paraphrase.

# Tagging
Tag every question on two axes.

Cognitive Level (type) — what kind of thinking the question requires:
  L1 Factual Recall        — directly stated facts
  L2 Conceptual            — definitions, role/purpose
  L3 Procedural            — sequences, ordered steps
  L4 Application           — apply rules to a concrete scenario / compute
  L5 Analytical/Diagnostic — violation, root-cause, what-if, debugging
  L6 Comparative/Design    — compare options, versions, trade-offs, overall/specific designs

Question Format (format) — the surface shape of the question:
  F1 Open-ended       — free-form prose answer expected
  F2 Short-answer     — a single term, value, or one-line answer expected
  F3 Yes/No + Justify — binary judgment followed by reasoning
  F4 Multiple Choice  — one correct option among plausible alternatives

# Materials
A question MAY reference one figure or table for richer reasoning (relationships, constraints, timing, or behaviors implied by the material). Do not ask simple visual lookups ("what is written here"). When used, set "material" to the exact FILENAME and refer to it only with generic phrasing ("According to the figure," / "Based on the table,") — never by number or filename. One material per question.

# Rules
1. Self-contained: a reader with AMBA background but no access to the source must understand the question. Never reference "the passage", "section X", "the author", "above", "below", etc.
2. Grounded: every question must be answerable from the given chunk and its materials. Do not invent signals, values, or rules not supported by the input.
3. Natural phrasing: write as a practicing engineer would — precise protocol terminology, no meta-language about the spec itself.
4. Quantity and Quality: generate as many questions as the chunk genuinely supports. Be thorough on content-rich chunks; do not pad with trivial, overly local, or strained questions. Return the blank result [] for structure-only content (TOC, index, headers without body, etc.).
5. Diversity: no two questions should ask essentially the same thing. Vary phrasing, angle, and difficulty.
6. Neutrality: no first/second person, no hedging ("maybe", "I think"), no commentary about the spec being unclear.

# Output
Return ONLY a JSON array, no surrounding prose or code fences:
[
  {"question": "...", "type": "L?", "format": "F?"},
  {"question": "...", "type": "L?", "format": "F?", "material": "FILENAME"}
]
"""

_USER_PROMPT_PREFIX = "<text_chunk>\n"
_USER_PROMPT_SUFFIX = (
    "\n</text_chunk>\n\nGenerate questions. Return ONLY the JSON array."
)


class QGenerator:
    """Generate questions from a single text chunk using GPT."""

    def __init__(
        self,
        model: str = "gpt-5.1",
        api_key: str | None = None,
    ) -> None:
        """Initialize QGenerator with OpenAI client."""
        self.model = model
        self._client = OpenAI(api_key=api_key)
        logger.info(f"QGenerator initialized with model={model}")

    def generate(
        self,
        chunk_path: str | Path,
        materials_dir: str | Path,
    ) -> list[dict]:
        """Generate questions for a single text chunk markdown file.

        Reads the chunk file, detects any <material:FILENAME> tags, loads the
        corresponding files from materials_dir, and calls the GPT API with the
        text (and images/tables as multimodal content where applicable).
        Returns a list of question dicts, each augmented with a source_chunk field.
        """
        chunk_path = Path(chunk_path)
        materials_dir = Path(materials_dir)

        text = chunk_path.read_text(encoding="utf-8")
        source_chunk = chunk_path.name
        logger.info(f"Generating questions for chunk: {source_chunk}")

        material_filenames = MATERIAL_TAG_PATTERN.findall(text)
        logger.debug(
            f"Found {len(material_filenames)} material references in {source_chunk}"
        )

        messages = self._build_messages(text, material_filenames, materials_dir)
        raw_questions = self._call_api(messages)

        questions = self._parse_response(raw_questions)
        for q in questions:
            q["source_chunk"] = source_chunk

        logger.info(f"Generated {len(questions)} questions for {source_chunk}")
        return questions

    def _build_messages(
        self,
        text: str,
        material_filenames: list[str],
        materials_dir: Path,
    ) -> list[dict]:
        """Build the OpenAI messages payload with text and optional material content."""
        user_content: list[dict] = build_material_content_blocks(
            material_filenames, materials_dir
        )
        user_content.append(
            {
                "type": "text",
                "text": _USER_PROMPT_PREFIX + text + _USER_PROMPT_SUFFIX,
            }
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _call_api(self, messages: list[dict]) -> str:
        """Call the OpenAI chat completion API and return the response text."""
        logger.debug(f"Calling OpenAI API with model={self.model}")
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def _parse_response(self, response_text: str) -> list[dict]:
        """Parse the JSON array from GPT response."""
        try:
            questions = json.loads(response_text.strip())
            if not isinstance(questions, list):
                logger.warning(
                    "GPT response is not a JSON array, returning empty list"
                )
                return []
            return questions
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse GPT response as JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            return []
