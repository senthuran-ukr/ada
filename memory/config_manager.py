import json
import os
import sys
from pathlib import Path

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

BASE_DIR    = get_base_dir()
CONFIG_DIR  = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "api_keys.json"

def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def config_exists() -> bool:
    return CONFIG_FILE.exists()

def save_api_keys(gemini_api_key: str) -> None:
    ensure_config_dir()

    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    data["gemini_api_key"] = gemini_api_key.strip()

    CONFIG_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

def load_api_keys() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to load api_keys.json: {e}")
        return {}

def get_gemini_key() -> str | None:
    return load_api_keys().get("gemini_api_key")


def get_ai_provider() -> str:
    """Return the live assistant provider: ``gemini`` or ``openai``."""
    raw = (
        os.environ.get("ADA_AI_PROVIDER")
        or load_api_keys().get("ai_provider")
        or "gemini"
    )
    return "openai" if str(raw).strip().lower() in {"openai", "chatgpt"} else "gemini"


def get_openai_key() -> str | None:
    """Read the OpenAI key from the environment; never persist it to config."""
    return os.environ.get("OPENAI_API_KEY") or None


def get_openai_model() -> str:
    return (
        os.environ.get("OPENAI_REALTIME_MODEL")
        or load_api_keys().get("openai_realtime_model")
        or "gpt-realtime-2.1"
    )


def get_openai_voice() -> str:
    return (
        os.environ.get("OPENAI_REALTIME_VOICE")
        or load_api_keys().get("openai_realtime_voice")
        or "marin"
    )


def save_ai_provider(provider: str) -> None:
    """Persist a provider choice without storing either provider's secret."""
    ensure_config_dir()
    data = load_api_keys()
    data["ai_provider"] = (
        "openai"
        if str(provider).strip().lower() in {"openai", "chatgpt"}
        else "gemini"
    )
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")

def is_configured() -> bool:
    key = get_gemini_key()
    return bool(key and len(key) > 15)


def get_assistant_name() -> str:
    """Return the configured assistant name, or 'ADA' if not set."""
    return load_api_keys().get("assistant_name", "ADA") or "ADA"


def get_user_name() -> str:
    """Return the configured user name for addressing."""
    return load_api_keys().get("user_name", "")


def save_assistant_config(assistant_name: str, user_name: str) -> None:
    """Persist assistant name and user name to config."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["assistant_name"] = assistant_name.strip() or "ADA"
    data["user_name"] = user_name.strip()
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_brief_enabled() -> bool:
    return load_api_keys().get("morning_brief_enabled", True)


def save_brief_enabled(enabled: bool) -> None:
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["morning_brief_enabled"] = enabled
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")
