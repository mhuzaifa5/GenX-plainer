"""Session and immutable case storage for GenXplainer."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_SAFE_ID = re.compile(r"^(sess|case|chat)_[a-f0-9]{32}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _validate_id(value: str, prefix: str) -> str:
    if not _SAFE_ID.fullmatch(value) or not value.startswith(f"{prefix}_"):
        raise ValueError(f"Invalid {prefix} ID: {value!r}")
    return value


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class CaseContext:
    session_id: str
    case_id: str
    chat_id: str
    session_dir: Path
    case_dir: Path

    def metadata(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "case_id": self.case_id,
            "chat_id": self.chat_id,
        }


def create_case(
    output_root: str | Path,
    source_image: str | Path,
    session_id: str | None = None,
) -> tuple[CaseContext, Path]:
    """Create a new immutable case and make it current for its session."""
    root = Path(output_root).expanduser().resolve() / "sessions"
    resolved_session_id = (
        _validate_id(session_id, "sess") if session_id else _new_id("sess")
    )
    case_id = _new_id("case")
    chat_id = _new_id("chat")
    session_dir = root / resolved_session_id
    case_dir = session_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=False)

    source = Path(source_image).expanduser().resolve()
    suffix = source.suffix.lower() or ".img"
    stored_image = case_dir / f"original{suffix}"
    shutil.copy2(source, stored_image)

    created_at = _now()
    session_path = session_dir / "session.json"
    if session_path.exists():
        session = json.loads(session_path.read_text(encoding="utf-8"))
    else:
        session = {
            "session_id": resolved_session_id,
            "created_at": created_at,
            "status": "active",
            "case_ids": [],
        }
    session["updated_at"] = created_at
    session["current_case_id"] = case_id
    session["case_ids"].append(case_id)
    _write_json(session_path, session)

    context = CaseContext(
        session_id=resolved_session_id,
        case_id=case_id,
        chat_id=chat_id,
        session_dir=session_dir,
        case_dir=case_dir,
    )
    _write_json(
        case_dir / "case.json",
        {
            **context.metadata(),
            "created_at": created_at,
            "status": "processing",
            "source_extension": suffix,
            "stored_image": stored_image.name,
        },
    )
    _write_json(
        case_dir / "chat_history.json",
        {**context.metadata(), "created_at": created_at, "messages": []},
    )
    return context, stored_image


def complete_case(context: CaseContext) -> None:
    """Mark a successfully processed case complete without altering older cases."""
    case_path = context.case_dir / "case.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case.update({"status": "complete", "completed_at": _now()})
    _write_json(case_path, case)


def close_session(output_root: str | Path, session_id: str) -> None:
    """Mark a session closed; deletion remains a separate retention-policy action."""
    validated = _validate_id(session_id, "sess")
    path = Path(output_root).expanduser().resolve() / "sessions" / validated / "session.json"
    if not path.is_file():
        raise FileNotFoundError(f"Session does not exist: {validated}")
    session = json.loads(path.read_text(encoding="utf-8"))
    session.update({"status": "closed", "closed_at": _now(), "updated_at": _now()})
    _write_json(path, session)
