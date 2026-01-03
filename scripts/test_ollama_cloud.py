import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from ollama import AsyncClient

async def test_connection(base_url, api_key):
    print(f"\n--- Testing Connection to: {base_url} ---")
    client = AsyncClient(
        host=base_url,
        headers={"Authorization": f"Bearer {api_key}"}
    )

    messages = [
        {"role": "user", "content": "Hello!"}
    ]

    try:
        async for chunk in await client.chat(
            model=settings.OLLAMA_MODEL,
            messages=messages,
            stream=True
        ):
            content = chunk.get("message", {}).get("content", "")
            print(content, end="", flush=True)
        print("\nSuccess!")
        return True
    except Exception as e:
        print(f"\nFailed: {e}")
        return False

async def main():
    api_key = settings.OLLAMA_API_KEY.get_secret_value() if settings.OLLAMA_API_KEY else ""
    
    # 1. Test Configured URL
    print(f"Configured OLLAMA_BASE_URL: {settings.OLLAMA_BASE_URL}")
    if await test_connection(settings.OLLAMA_BASE_URL, api_key):
        print("Configured URL works.")
    else:
        print("Configured URL failed.")
        
        # 2. Test Default/Doc URL if different
        default_url = "https://ollama.com"
        if settings.OLLAMA_BASE_URL != default_url:
            print(f"\nRetrying with default documentation URL: {default_url}")
            if await test_connection(default_url, api_key):
                print("Default URL works! Please update your OLLAMA_BASE_URL.")
            else:
                print("Default URL also failed.")

if __name__ == "__main__":
    asyncio.run(main())
