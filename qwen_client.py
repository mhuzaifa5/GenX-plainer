"""Send a GenXplainer prompt packet to the local vLLM Qwen server."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from prompt_packet import PromptPacket


@dataclass(frozen=True)
class QwenResult:
    response: dict[str, object]
    raw_text: str
    model: str

    def metadata(self) -> dict[str, object]:
        return {"model": self.model, "response": self.response}


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
        raise ValueError("Qwen response must be a JSON object")
    return value


def query_qwen(
    packet: PromptPacket,
    base_url: str = "http://127.0.0.1:8080/v1",
    model: str = "Qwen/Qwen3.5-9B",
    max_tokens: int = 1200,
) -> QwenResult:
    """Call the OpenAI-compatible vLLM endpoint with prompt and images."""
    content: list[dict[str, object]] = [
        {"type": "text", "text": packet.prompt_text}
    ]
    for attachment in packet.attachment_paths:
        path = Path(attachment).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Qwen attachment does not exist: {path}")
        content.append(
            {"type": "image_url", "image_url": {"url": _data_uri(path)}}
        )

    client = OpenAI(base_url=base_url.rstrip("/"), api_key="dummy")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    message = completion.choices[0].message
    raw_text = message.content or ""
    if not raw_text.strip():
        raise RuntimeError("Qwen returned an empty response")
    return QwenResult(
        response=_extract_json(raw_text), raw_text=raw_text, model=model
    )


def save_qwen_result(result: QwenResult, output_dir: str | Path) -> dict[str, Path]:
    """Save normalized JSON and raw model text for traceability."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    response_path = directory / "qwen_response.json"
    raw_path = directory / "qwen_response_raw.txt"
    response_path.write_text(
        json.dumps(result.response, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    raw_path.write_text(result.raw_text + "\n", encoding="utf-8")
    return {"response": response_path, "raw": raw_path}
