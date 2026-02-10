import os
from dotenv import load_dotenv
from openai import OpenAI

# Load your .env file
load_dotenv()

# Retrieve the key from .env
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ No API key found.")
else:
    # Print the first 5 characters to avoid leaking the full key in logs
    print(f"✅ API successfully loaded from .env! (Starts with: {api_key[:5]}...)")
    
    # Finall Test that keys are valid with a test request
    try:
        client = OpenAI(api_key=api_key)
        # We just ask for a list of models to verify the key is valid
        client.models.list()
        print("✅ Connection successful! OpenAI key is active and working.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")