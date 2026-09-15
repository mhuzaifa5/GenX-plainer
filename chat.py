"""Module 8: grounded, case-scoped chat with the local vLLM Qwen server."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = """You are a case-grounded chest X-ray assistant.
Use only the supplied case packet, initial report, images, and chat history.
Keep image observations, model outputs, attribution, and clinical interpretation separate.
Attribution shows model influence; it does not identify a recognized finding, cause,
disease location, severity, or probability. Model scores are not clinical probabilities.
Do not accept assumptions in a question as facts or claim overlap unless it is supplied.
If evidence is insufficient, say: "The supplied case evidence does not establish this."
Do not give a definitive diagnosis or treatment recommendation.

Return only valid JSON with this shape:
{
  "answer": "",
  "case_evidence_used": [],
  "general_information": [],
  "limitations": []
}
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_json(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Qwen did not return a JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Qwen chat response must be a JSON object")
    return value


def _validate_response(value: dict[str, object]) -> None:
    required = {
        "answer": str,
        "case_evidence_used": list,
        "general_information": list,
        "limitations": list,
    }
    for field, expected_type in required.items():
        if not isinstance(value.get(field), expected_type):
            raise ValueError(
                f"Qwen chat response field {field!r} must be {expected_type.__name__}"
            )


def _touch_session(case_dir: Path, timestamp: str) -> None:
    session_path = case_dir.parent.parent / "session.json"
    if session_path.is_file():
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["updated_at"] = timestamp
        _write_json(session_path, session)


def _load_case(case_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    case_path = case_dir / "case.json"
    history_path = case_dir / "chat_history.json"
    if not case_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Not a complete GenXplainer case: {case_dir}")
    return (
        json.loads(case_path.read_text(encoding="utf-8")),
        json.loads(history_path.read_text(encoding="utf-8")),
    )


def _case_images(case_dir: Path, case: dict[str, object]) -> list[Path]:
    stored_name = case.get("stored_image")
    if not isinstance(stored_name, str):
        raise ValueError("case.json does not contain stored_image")
    images = [case_dir / stored_name]
    overlay = case_dir / "attribution_prediction_segmentation_overlay.png"
    images.append(overlay)
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"Required case image is missing: {image}")
    return images


def chat_once(
    case_dir: str | Path,
    question: str,
    base_url: str = "http://127.0.0.1:8080/v1",
    model: str = "Qwen/Qwen3.5-9B",
    max_tokens: int = 800,
    history_messages: int = 6,
    include_images: bool = True,
) -> dict[str, object]:
    """Ask one grounded question and append the exchange to this case only."""
    directory = Path(case_dir).expanduser().resolve()
    case, history = _load_case(directory)
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Question cannot be empty")
    if len(clean_question) > 4000:
        raise ValueError("Question is too long (maximum 4000 characters)")

    packet_path = directory / "prompt_packet" / "prompt_packet.json"
    initial_path = directory / "qwen_response.json"
    if not packet_path.is_file() or not initial_path.is_file():
        raise FileNotFoundError(
            "Chat requires prompt_packet/prompt_packet.json and qwen_response.json"
        )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    frozen_prompt = packet.get("prompt")
    if not isinstance(frozen_prompt, str):
        raise ValueError("Prompt packet does not contain a text prompt")

    messages: list[dict[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    context_text = (
        "FROZEN CASE PACKET:\n"
        + frozen_prompt
        + "\n\nINITIAL GROUNDED REPORT:\n"
        + json.dumps(initial, ensure_ascii=False)
    )
    context_content: list[dict[str, object]] = [
        {"type": "text", "text": context_text}
    ]
    if include_images:
        for image in _case_images(directory, case):
            context_content.append(
                {"type": "image_url", "image_url": {"url": _data_uri(image)}}
            )
    messages.append({"role": "user", "content": context_content})
    messages.append(
        {
            "role": "assistant",
            "content": "Case context loaded. I will answer only from this case.",
        }
    )

    prior = history.get("messages", [])
    if not isinstance(prior, list):
        raise ValueError("chat_history.json messages must be a list")
    for item in prior[-max(0, history_messages) :]:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            messages.append(
                {"role": item["role"], "content": str(item.get("content", ""))}
            )
    messages.append({"role": "user", "content": clean_question})

    client = OpenAI(base_url=base_url.rstrip("/"), api_key="dummy")
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = completion.choices[0].message.content or ""
    if not raw.strip():
        raise RuntimeError("Qwen returned an empty chat response")
    response = _extract_json(raw)
    _validate_response(response)

    timestamp = _now()
    prior.extend(
        [
            {"role": "user", "content": clean_question, "created_at": timestamp},
            {
                "role": "assistant",
                "content": json.dumps(response, ensure_ascii=False),
                "created_at": timestamp,
            },
        ]
    )
    history["messages"] = prior
    history["updated_at"] = timestamp
    _write_json(directory / "chat_history.json", history)
    _touch_session(directory, timestamp)
    (directory / "latest_chat_response.json").write_text(
        json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--question", help="Ask once; omit for interactive chat")
    parser.add_argument(
        "--base-url",
        default=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8080/v1"),
    )
    parser.add_argument(
        "--model", default=os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-9B")
    )
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--history-messages", type=int, default=6)
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Use frozen evidence/report without resending images",
    )
    args = parser.parse_args()

    def ask(question: str) -> None:
        response = chat_once(
            args.case_dir,
            question,
            args.base_url,
            args.model,
            args.max_tokens,
            args.history_messages,
            not args.no_images,
        )
        print(json.dumps(response, indent=2, ensure_ascii=False))

    if args.question is not None:
        ask(args.question)
        return

    print("Grounded case chat. Type 'exit' to finish.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            ask(question)


if __name__ == "__main__":
    main()
