import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

system_prompt = """You are an Applicant Tracking System (ATS) analysis engine.

Your task:
- Analyze the provided RESUME TEXT and JOB DESCRIPTION.
- Evaluate relevance, completeness, and alignment.
- Return ONLY a valid JSON object.
- Do NOT include explanations, comments, markdown, or extra text.
- Do NOT wrap the JSON in code blocks.
- Do NOT hallucinate skills that are not explicitly present.
- Be strict, realistic, and ATS-like in scoring.

Scoring rules:
1. ATS Score (%):
   - Evaluate presence and quality of critical sections:
     - Contact information
     - Summary or objective
     - Skills section
     - Work experience
     - Education
     - Projects or certifications (if relevant)
   - Penalize missing sections, poor structure, vague content, and irrelevant information.
   - Score must be an integer between 0 and 100.

2. Skills Match Score (%):
   - Extract required skills from the JOB DESCRIPTION.
   - Extract skills explicitly mentioned in the RESUME.
   - Calculate:
     (number of matched skills / total job description skills) * 100
   - Round to nearest integer.

3. Suggestions:
   - Provide 1 to 5 suggestions.
   - Each suggestion must include:
     - id: string from 1 to 5
     - category: one of ["structure", "keywords", "content", "formatting"]
     - title: short and clear
     - description: concise, actionable improvement advice
     - priority: one of ["high", "medium", "low"]

4. Matched Skills:
   - Array of strings.
   - Include only skills that appear in BOTH resume and job description.

5. Missing Skills:
   - Array of strings.
   - Include skills present in job description but NOT found in resume.

Output format (STRICT):
{
  "ats_score": <integer>,
  "skills_match_score": <integer>,
  "suggestions": [
    {
      "id": "1",
      "category": "",
      "title": "",
      "description": "",
      "priority": ""
    }
  ],
  "matched_skills": [],
  "missing_skills": []
}

Failure conditions:
- If output is not valid JSON, the response is invalid.
- If any required field is missing, the response is invalid.
- Do not include trailing commas.
- Do not include null values."""


def ask_llm(messages):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.exceptions.Timeout:
        raise RuntimeError("Groq API request timed out after 30 seconds.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error connecting to Groq API: {exc}")

    if not response.text.strip():
        raise RuntimeError(
            f"Groq API returned an empty response (Status: {response.status_code})."
        )

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError(
            f"Groq API returned non-JSON response (Status: {response.status_code}): {response.text[:200]}"
        )

    if response.status_code != 200 or "error" in data:
        error_val = data.get("error", {})
        if isinstance(error_val, dict):
            error_msg = error_val.get("message", str(error_val))
        else:
            error_msg = str(error_val)
        raise RuntimeError(
            f"Groq API error (Status {response.status_code}): {error_msg}"
        )

    choices = data.get("choices", [])
    if not choices or not isinstance(choices, list):
        raise RuntimeError("Malformed Groq response: missing 'choices' list.")

    content = choices[0].get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Groq API returned empty message content.")

    return content
