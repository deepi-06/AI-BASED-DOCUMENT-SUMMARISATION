
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Test Groq
def test_groq():
    try:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[
                {"role": "user", "content": "Say Hello in one sentence."}
            ],
            max_completion_tokens=100
        )

        print("GROQ: SUCCESS")
        print(response.choices[0].message.content)

    except Exception as e:
        print("GROQ: FAILED")
        print(type(e).__name__, str(e)[:300])


# 2. Test Gemini
def test_gemini():
    try:
        from google import genai

        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents="Say Hello in one sentence."
        )

        print("GEMINI: SUCCESS")
        print(response.text)

    except Exception as e:
        print("GEMINI: FAILED")
        print(type(e).__name__, str(e)[:300])


# 3. Test OpenAI
def test_openai():
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "user", "content": "Say Hello in one sentence."}
            ],
            max_completion_tokens=100
        )

        print("OPENAI: SUCCESS")
        print(response.choices[0].message.content)

    except Exception as e:
        print("OPENAI: FAILED")
        print(type(e).__name__, str(e)[:300])


print("\n===== API TESTING STARTED =====\n")

for name, key in [
    ("GROQ", "GROQ_API_KEY"),
    ("GEMINI", "GEMINI_API_KEY"),
    ("OPENAI", "OPENAI_API_KEY")
]:
    if not os.getenv(key):
        print(f"{name}: API KEY MISSING")

test_groq()
test_gemini()
test_openai()

print("\n===== API TESTING COMPLETED =====")
