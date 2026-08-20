"""Single place that resolves which LLM provider to use, based on
config.yaml's llm.<provider>.active boolean flags. Both rag_pipeline.py
(report synthesis) and chat_qa.py (follow-up Q&A) call get_llm() and
has_api_key() from here, so switching providers is a config-only change
(flip which provider has active: true), not a code change.
"""
import os
import requests
import yaml
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _CONFIG = yaml.safe_load(_f)

_LLM_CONFIG = _CONFIG["llm"]

_REQUIRED_ENV_VAR = {
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _resolve_active_provider() -> str:
    active = [name for name in _REQUIRED_ENV_VAR if _LLM_CONFIG.get(name, {}).get("active")]
    if len(active) == 0:
        raise ValueError(
            "No LLM provider is active in config.yaml (llm.gemini/groq/openai all "
            "have active: false). Set exactly one to active: true."
        )
    if len(active) > 1:
        raise ValueError(
            f"Multiple LLM providers are active in config.yaml: {active}. "
            "Set exactly one to active: true, and the rest to false."
        )
    return active[0]


PROVIDER = _resolve_active_provider()


def has_api_key() -> bool:
    return bool(os.environ.get(_REQUIRED_ENV_VAR[PROVIDER]))


class DirectGeminiLLM:
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature
        
    def invoke(self, prompt):
        api_key = os.getenv("GOOGLE_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": str(prompt)}]}],
            "generationConfig": {"temperature": self.temperature}
        }
        res = requests.post(url, json=payload)
        data = res.json()
        
        class FakeResponse:
            pass
            
        ret = FakeResponse()
        if "candidates" in data:
            ret.content = data["candidates"][0]["content"]["parts"][0]["text"]
            return ret
        else:
            raise Exception(f"Gemini API Error: {data}")


def get_llm():
    """Returns a LangChain chat model instance for whichever provider has
    active: true in config.yaml."""
    cfg = _LLM_CONFIG[PROVIDER]
    if PROVIDER == "groq":
        return ChatGroq(model=cfg["model"], temperature=cfg["temperature"])
    elif PROVIDER == "gemini":
        return DirectGeminiLLM(model=cfg["model"], temperature=cfg["temperature"])
    elif PROVIDER == "openai":
        return ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])
    raise ValueError(f"Unknown LLM provider resolved: {PROVIDER!r}")
