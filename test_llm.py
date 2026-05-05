from openai import OpenAI

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(override=False)

import os

base_url = os.getenv("OPENAI_BASE_URL", "https://openai.rc.asu.edu/v1").strip()
api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
if not api_key:
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Set it in your shell or a local `.env` file.\n"
        "Example: export OPENAI_API_KEY='...'\n"
        "Optional: export OPENAI_BASE_URL='https://openai.rc.asu.edu/v1'"
    )

client = OpenAI(base_url=base_url, api_key=api_key)

response = client.chat.completions.create(
    model="qwen3-30b-a3b-thinking-2507", messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)


# gemma4-31b-it
# llama4-scout-17b
# qwen3-30b-a3b-thinking-2507