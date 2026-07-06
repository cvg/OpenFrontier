import os
from typing import Optional


GOOGLE_API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def get_google_api_key(required: bool = True) -> Optional[str]:
    for env_var in GOOGLE_API_KEY_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return value

    if required:
        names = " or ".join(GOOGLE_API_KEY_ENV_VARS)
        raise RuntimeError(f"Set {names} in the environment before using Gemini.")

    return None


def mask_api_key(api_key: Optional[str]) -> str:
    if not api_key:
        return "<unset>"

    if len(api_key) <= 8:
        return "<redacted>"

    return f"{api_key[:4]}...{api_key[-4:]}"
