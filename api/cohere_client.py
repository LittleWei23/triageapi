"""
Creates and exports a reusable Cohere client.

The API key is loaded from the .env file.
"""

import os

import cohere
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(
    os.getenv("COHERE_API_KEY")
)