import os
import requests
from dotenv import load_dotenv

# Load the API key from your .env file
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

print("Asking Google for available models...\n")

# Call the exact Google API endpoint
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)

if response.status_code == 200:
    models = response.json().get("models", [])
    print("✅ THESE ARE YOUR WORKING MODEL NAMES:\n" + "-"*40)
    for m in models:
        # Only print models that support text generation
        if "generateContent" in m.get("supportedGenerationMethods", []):
            # Remove the "models/" prefix so it's ready to paste into LangChain
            clean_name = m["name"].replace("models/", "")
            print(f'"{clean_name}"')
else:
    print("❌ Failed to connect:", response.text)