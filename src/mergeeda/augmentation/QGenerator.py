"""Question generation module using OpenAI GPT for text chunk question generation."""

import json
import logging
from pathlib import Path

from openai import OpenAI

from mergeeda.utils import MATERIAL_TAG_PATTERN, build_material_content_blocks

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior AMBA protocol expert (AXI, APB, AHB, CHI, etc.) with deep implementation and verification experience. Your task is to generate high-quality questions from a given AMBA specification chunk.

# Input
You will receive:
- A text chunk from the specification.
- Optionally, figures (provided inline as images) and/or tables (provided as labeled text blocks in the form `[Table: FILENAME]`).
- Figures and tables are referenced in the text as `<material:FILENAME>`.

# Goal
Produce questions that a practicing engineer would realistically ask while reading, implementing, or verifying against the specification.

# Question Types
Classify each question by the kind of reasoning it requires:
- L1 Factual Recall          — directly stated facts
- L2 Conceptual              — definitions, roles, purpose
- L3 Procedural              — sequences, ordered steps, handshakes
- L4 Application             — apply rules to a concrete scenario or compute a result
- L5 Analytical / Diagnostic — protocol violations, root-cause analysis, what-if, debugging
- L6 Comparative / Design    — compare options, versions, or trade-offs; design rationale

# Use of Materials
- A question MAY reference at most ONE figure or table to enable richer reasoning about relationships, timing, constraints, or implied behaviors.
- Do NOT ask trivial visual-lookup questions (e.g., "what value is shown here").
- When a material is used, set `"material"` to the exact FILENAME, and refer to it in the question text only with generic phrasing such as "According to the figure," or "Based on the table," — never by figure/table number or filename.

# Rules
1. **Self-contained**: A reader with general AMBA background but no access to the source must fully understand the question. Never reference "the passage", "this section", "the author", "above", "below", etc.
2. **Grounded**: Every question must be answerable from the provided chunk (and its referenced material, if any). Do not invent signals, values, or rules not supported by the input.
3. **Natural phrasing**: Write as a practicing engineer would — precise protocol terminology, no meta-language about the specification itself.
4. **Quantity matches substance**: Generate as many questions as the chunk genuinely supports. Be thorough on content-rich chunks; do not pad with trivial, overly local, or strained questions. Return `[]` for structure-only content (TOC, index, standalone headers, etc.).
5. **Diversity**: No two questions should test essentially the same point. Vary angle, difficulty, and phrasing.
6. **Neutrality**: No first- or second-person voice, no hedging ("maybe", "I think"), no commentary about the specification being unclear or incomplete.

# Output Format
Return ONLY a JSON array. No surrounding prose, no explanations, no code fences.

Each element must follow this schema:
[
  {"question": "...", "type": "L1" | "L2" | "L3" | "L4" | "L5" | "L6"},
  {"question": "...", "type": "L?", "material": "FILENAME"},
  ...
]

The `"material"` field is included only when the question references a figure or table.
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
