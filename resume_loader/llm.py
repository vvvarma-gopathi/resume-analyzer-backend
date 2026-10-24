
import json
import logging
import os

import requests


logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT = (10, 60)


SYSTEM_PROMPT = """
You are an Applicant Tracking System (ATS) analysis engine.

Analyze the provided resume text against the job description.

RULES:
- Be strict, realistic, and evidence-based.
- Never invent skills, experience, qualifications, or achievements.
- Return only a valid JSON object.
- Do not include Markdown, comments, or explanatory text outside JSON.
- Every required field must be present.
- Never use null values.

SCORING:

1. ats_score:
   - Integer between 0 and 100.
   - Evaluate contact information, summary/objective, skills,
     work experience, education, projects, and certifications.
   - Penalize missing sections, poor structure, vague content,
     and irrelevant information.
   - Evaluate the resume's overall quality and ATS readability.

2. skills_match_score:
   - Integer between 0 and 100.
   - Identify distinct skills explicitly required by the job description.
   - Match them against skills explicitly evidenced in the resume.
   - Calculate matched required skills / total distinct required skills * 100.
   - If no identifiable skills are required, return 0.

3. suggestions:
   - Return between 1 and 5 actionable suggestions.
   - Each suggestion must contain:
       id: string, such as "1"
       category: "structure", "keywords", "content", or "formatting"
       title: short string
       description: concise, actionable advice
       priority: "high", "medium", or "low"

4. matched_skills:
   - Array of strings.
   - Include only skills supported by the resume and required by
     the job description.
   - Do not include skills merely inferred from a job title.

5. missing_skills:
   - Array of strings.
   - Include required job-description skills not evidenced in the resume.
   - Do not classify a skill as missing if it is explicitly evidenced
     in the resume, even if synonyms are used.

OUTPUT FORMAT:

{
    "ats_score": 75,
    "skills_match_score": 60,
    "suggestions": [
        {
            "id": "1",
            "category": "keywords",
            "title": "Add relevant keywords",
            "description": "Include relevant job requirements supported by your experience.",
            "priority": "high"
        }
    ],
    "matched_skills": ["Python"],
    "missing_skills": ["Docker"]
}

The numbers and skills above are illustrative, not actual assessment results.
"""


REQUIRED_FIELDS = {
    "ats_score",
    "skills_match_score",
    "suggestions",
    "matched_skills",
    "missing_skills",
}

VALID_CATEGORIES = {
    "structure",
    "keywords",
    "content",
    "formatting",
}

VALID_PRIORITIES = {
    "high",
    "medium",
    "low",
}


class GroqAPIError(RuntimeError):
    """Raised when Groq cannot complete a request."""


class InvalidLLMResponseError(RuntimeError):
    """Raised when the model returns an invalid analysis."""


def _validate_analysis(content: str) -> dict:
    """
    Parse and validate the model's JSON response.
    """

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidLLMResponseError(
            "The AI returned invalid JSON."
        ) from exc

    if not isinstance(result, dict):
        raise InvalidLLMResponseError(
            "The AI response must be a JSON object."
        )

    missing_fields = REQUIRED_FIELDS - result.keys()

    if missing_fields:
        raise InvalidLLMResponseError(
            f"Missing required fields: {sorted(missing_fields)}"
        )

    # Validate numeric scores. bool is a subclass of int in Python,
    # so explicitly reject booleans.
    for field in ("ats_score", "skills_match_score"):
        score = result[field]

        if (
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 100
        ):
            raise InvalidLLMResponseError(
                f"{field} must be an integer between 0 and 100."
            )

    # Validate skill lists.
    for field in ("matched_skills", "missing_skills"):
        skills = result[field]

        if (
            not isinstance(skills, list)
            or not all(
                isinstance(skill, str) and skill.strip()
                for skill in skills
            )
        ):
            raise InvalidLLMResponseError(
                f"{field} must be a list of non-empty strings."
            )

    # Validate suggestions.
    suggestions = result["suggestions"]

    if (
        not isinstance(suggestions, list)
        or not 1 <= len(suggestions) <= 5
    ):
        raise InvalidLLMResponseError(
            "Suggestions must contain between 1 and 5 items."
        )

    required_suggestion_fields = {
        "id",
        "category",
        "title",
        "description",
        "priority",
    }

    for index, suggestion in enumerate(suggestions, start=1):
        if not isinstance(suggestion, dict):
            raise InvalidLLMResponseError(
                f"Suggestion {index} must be a JSON object."
            )

        missing = required_suggestion_fields - suggestion.keys()

        if missing:
            raise InvalidLLMResponseError(
                f"Suggestion {index} is missing fields: "
                f"{sorted(missing)}"
            )

        for field in (
            "id",
            "category",
            "title",
            "description",
            "priority",
        ):
            if (
                not isinstance(suggestion[field], str)
                or not suggestion[field].strip()
            ):
                raise InvalidLLMResponseError(
                    f"Suggestion {index}: {field} must be a "
                    "non-empty string."
                )

        if suggestion["category"] not in VALID_CATEGORIES:
            raise InvalidLLMResponseError(
                f"Suggestion {index} has an invalid category."
            )

        if suggestion["priority"] not in VALID_PRIORITIES:
            raise InvalidLLMResponseError(
                f"Suggestion {index} has an invalid priority."
            )

    # Discard unexpected top-level fields to preserve the API contract.
    return {
        "ats_score": result["ats_score"],
        "skills_match_score": result["skills_match_score"],
        "suggestions": suggestions,
        "matched_skills": result["matched_skills"],
        "missing_skills": result["missing_skills"],
    }


def ask_llm(messages: list[dict]) -> str:
    """
    Send resume-analysis messages to Groq.

    Returns:
        A validated JSON string, compatible with existing Django
        code that expects ask_llm() to return a string.
    """

    if not GROQ_API_KEY:
        raise GroqAPIError(
            "GROQ_API_KEY environment variable is not configured."
        )

    if not isinstance(messages, list) or not messages:
        raise ValueError(
            "messages must be a non-empty list of chat messages."
        )

    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError(
                f"Invalid chat message at index {index}."
            )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            *messages,
        ],
        "response_format": {
            "type": "json_object",
        },
        "temperature": 0.2,
        "max_completion_tokens": 2500,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=GROQ_TIMEOUT,
        )

    except requests.exceptions.Timeout as exc:
        logger.warning("Groq API request timed out.")

        raise GroqAPIError(
            "The AI service timed out. Please try again."
        ) from exc

    except requests.exceptions.RequestException as exc:
        logger.exception("Network error while contacting Groq.")

        raise GroqAPIError(
            "Unable to connect to the AI service."
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        logger.error(
            "Groq returned non-JSON data with HTTP status %s.",
            response.status_code,
        )

        raise GroqAPIError(
            "The AI service returned an invalid response."
        ) from exc

    if not response.ok or "error" in data:
        error = data.get("error", {})

        if isinstance(error, dict):
            error_message = error.get(
                "message",
                "Unknown Groq API error.",
            )
            error_code = error.get("code")
        else:
            error_message = str(error)
            error_code = None

        # Log details on the server, not in the public API response.
        logger.error(
            "Groq API error: status=%s code=%s message=%s",
            response.status_code,
            error_code,
            error_message,
        )

        if response.status_code == 401:
            raise GroqAPIError(
                "The AI service authentication failed. "
                "Check the configured API key."
            )

        if response.status_code == 403:
            raise GroqAPIError(
                "The AI service denied access to this model."
            )

        if response.status_code == 404:
            raise GroqAPIError(
                "The configured AI model was not found or is "
                "unavailable to this account."
            )

        if response.status_code == 429:
            raise GroqAPIError(
                "The AI service is rate-limited or has reached "
                "its usage limit. Please try again later."
            )

        if response.status_code >= 500:
            raise GroqAPIError(
                "The AI provider is temporarily unavailable."
            )

        raise GroqAPIError(
            f"The AI request failed with HTTP "
            f"{response.status_code}."
        )

    choices = data.get("choices")

    if not isinstance(choices, list) or not choices:
        raise InvalidLLMResponseError(
            "The AI response contains no completion choices."
        )

    choice = choices[0]

    if choice.get("finish_reason") == "length":
        raise InvalidLLMResponseError(
            "The AI response was truncated. Reduce the input size "
            "or increase the output-token limit."
        )

    content = choice.get("message", {}).get("content")

    if not isinstance(content, str) or not content.strip():
        raise InvalidLLMResponseError(
            "The AI returned empty analysis content."
        )

    validated_result = _validate_analysis(content)

    # Preserve the string return type expected by AnalyzeResume.
    return json.dumps(
        validated_result,
        ensure_ascii=False,
    )
