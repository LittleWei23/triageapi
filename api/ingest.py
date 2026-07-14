"""
Reads support tickets from a JSONL file,
combines subject and body into one searchable document,
embeds them using Cohere Embed,
and stores everything in ChromaDB.
"""

import json
import chromadb

from cohere_client import co

# ---------------------------------------
# Connect to ChromaDB
# ---------------------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="support_tickets"
)

# ---------------------------------------
# Read JSONL dataset
# ---------------------------------------

tickets = []

with open("data/tickets.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        tickets.append(json.loads(line))

print(f"Loaded {len(tickets)} tickets.")

# ---------------------------------------
# Create searchable documents
# ---------------------------------------

documents = [
    f"Subject: {ticket['subject']}\n\nBody:\n{ticket['body']}"
    for ticket in tickets
]

# ---------------------------------------
# Generate embeddings
# ---------------------------------------

print("Generating embeddings...")

response = co.embed(
    texts=documents,
    model="embed-english-v3.0",
    input_type="search_document"
)

embeddings = response.embeddings

# ---------------------------------------
# Store in ChromaDB
# ---------------------------------------

collection.add(
    ids=[ticket["id"] for ticket in tickets],
    documents=documents,
    embeddings=embeddings,
    metadatas=[
        {
            "subject": ticket["subject"],
            "category": ticket["category"],
            "priority": ticket["priority"]
        }
        for ticket in tickets
    ]
)

print("Successfully stored tickets in ChromaDB.")