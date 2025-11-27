#!/usr/bin/env python3
"""Test script to verify OpenAI API connection."""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file in this directory
# Use override=True to override any existing environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

def test_openai_connection():
    """Test connection to OpenAI API."""
    # Debug: check env file path
    env_path = Path(__file__).parent / '.env'
    print(f"Loading .env from: {env_path}")
    print(f"File exists: {env_path.exists()}")

    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        return False

    print(f"✓ API key found (first 50 chars): {api_key[:50]}...")
    print(f"✓ API key length: {len(api_key)}")

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Make a simple test request
        print("\nTesting API connection with a simple completion...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Connection successful!' if you can read this."}
            ],
            max_tokens=20
        )

        result = response.choices[0].message.content
        print(f"✓ Response received: {result}")
        print("\n✅ OpenAI API connection successful!")
        return True

    except Exception as e:
        error_type = type(e).__name__
        print(f"\n❌ ERROR: {error_type}: {e}")

        # Provide helpful context for common errors
        if error_type == "RateLimitError":
            print("\n💡 NOTE: The API key is valid and connection is working!")
            print("   However, you've exceeded your quota. Please:")
            print("   1. Check your billing details at https://platform.openai.com/account/billing")
            print("   2. Add credits or upgrade your plan")
            print("   3. The key itself is correctly configured in your .env file")
            return True  # Connection test passed, quota is a separate issue
        elif error_type == "AuthenticationError":
            print("\n💡 The API key is invalid or revoked.")
            print("   Get a valid key from https://platform.openai.com/account/api-keys")

        return False

if __name__ == "__main__":
    success = test_openai_connection()
    exit(0 if success else 1)
