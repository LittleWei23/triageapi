"""
FastAPI application providing a ticket triage endpoint.

Workflow:

Incoming Ticket
        ↓
Embed Query
        ↓
Retrieve Top-10 Similar Tickets
        ↓
Send Context to Cohere Command
        ↓
Return Classification + Suggested Response
"""

import json
import chromadb

from fastapi import FastAPI
from pydantic import BaseModel

from cohere_client import co
from prompts import SYSTEM_PROMPT

app = FastAPI(
    title="Support Ticket Triage API"
)

# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    "support_tickets"
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class TicketRequest(BaseModel):
    ticket: str


# --------------------------------------------------
# POST /v1/triage
# --------------------------------------------------

@app.post("/v1/triage")
def triage(request: TicketRequest):

    # Step 1: Embed incoming ticket

    query_embedding = co.embed(
        texts=[request.ticket],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    # Step 2: Retrieve similar tickets

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    similar_ids = results["ids"][0]
    similar_docs = results["documents"][0]
    if not similar_ids or not similar_docs:
    # Handle the empty case safely (e.g., return early or raise a controlled error)
        return {"status": "success", "context": "No context available"}

    # Step 3: Build retrieval context

    context = ""

    for ticket_id, document in zip(similar_ids, similar_docs):

        context += f"""
        Ticket ID: {ticket_id}

        {document}

        ----------------------------------------
        """
    # Step 4: Build prompt

    prompt = f"""
New Customer Ticket

{request.ticket}

----------------------------------------

Similar Historical Tickets

{context}

Using the historical tickets as reference,
classify the ticket and generate a response.

Return ONLY JSON.
"""

    # Step 5: Call Cohere Command

    response = co.chat(
        model="command-a-03-2025",
        preamble=SYSTEM_PROMPT,
        message=prompt,
        # This tells Cohere to strictly output a JSON object
        response_format={"type": "json_object"}
    )

    # Step 6: Convert JSON string to dictionary

    # print("========== RAW LLM RESPONSE ==========")
    # print(response.text)
    # print("======================================")

    try:
        output = json.loads(response.text)
    except json.JSONDecodeError:
        # Fallback in case there are markdown code fences like ```json ... ```
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        try:
            output = json.loads(clean_text)
        except json.JSONDecodeError as e:
            # If it still fails, log the bad text and raise a clean FastAPI error
            from fastapi import HTTPException
            raise HTTPException(
                status_code=500, 
                detail=f"LLM failed to output valid JSON. Raw text: {response.text[:100]}"
            )

    # Ensure retrieved IDs are returned
    output["similar_ticket_ids"] = similar_ids
    return output