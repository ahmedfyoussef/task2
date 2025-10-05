
"""
Minimal Retrieval-Augmented Answering Service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import re


app = FastAPI(title="Retrieval QA Service")


model = SentenceTransformer('all-MiniLM-L6-v2')

# Defining the knowledge base

documents = [
    "Our company offers 15 vacation days per year for all employees.",
    "The healthcare plan includes dental and vision coverage with $20 copays.",
    "Employees can work remotely up to 3 days per week with manager approval.",
    "The 401(k) matching program provides 4% company match after 1 year of service.",
    "Paid time off accrues at a rate of 1.25 days per month for full-time employees.",
    "The parental leave policy offers 12 weeks of paid leave for new parents.",
    "Performance reviews are conducted bi-annually in June and December.",
    "The office opens at 8:00 AM and closes at 6:00 PM on weekdays.",
    "Parking validation is available for employees working in the downtown office.",
    "Professional development budget is $1,000 per employee per fiscal year.",
    "The dress code is business casual Monday through Thursday, casual on Fridays.",
    "All employees receive company-issued laptops and mobile phones for work purposes.",
    "Travel expenses are reimbursed within 14 days of submission with proper receipts.",
    "The company provides free lunch on Wednesdays and stocked snacks daily.",
    "New hires undergo a 30-day onboarding program with mentorship support."
]

# Encode the documents once at startup to avoid repeated computation
document_embeddings = model.encode(documents)

# Define request and response models using Pydantic
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    top_snippets: List[str]
    search_method: str
    processing_time: float

# Guardrail: Block inappropriate or sensitive topics
DENYLIST_KEYWORDS = [
    "salary", "compensation", "pay rate", "bonus", "executive",
    "lawsuit", "legal", "layoff", "termination"
]

def check_guardrail(question: str) -> bool:
    """Block sensitive HR-related topics"""
    question_lower = question.lower()
    for keyword in DENYLIST_KEYWORDS:
        if keyword in question_lower:
            return False
    return True

# Search method using cosine similarity
def search_cosine(question: str, top_k: int = 3) -> List[str]:
    """Search for the top-k most relevant documents using cosine similarity"""
    start_time = time.time()


    question_embedding = model.encode([question])


    similarities = cosine_similarity(question_embedding, document_embeddings)[0]


    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_snippets = [documents[i] for i in top_indices]

    return top_snippets, time.time() - start_time

# Search method using dot product (faster, but less effective than cosine similarity)
def search_dot_product(question: str, top_k: int = 3) -> List[str]:
    """Search for the top-k most relevant documents using dot product"""
    start_time = time.time()


    question_embedding = model.encode([question])


    dot_products = np.dot(question_embedding, document_embeddings.T)[0]


    top_indices = np.argsort(dot_products)[-top_k:][::-1]
    top_snippets = [documents[i] for i in top_indices]

    return top_snippets, time.time() - start_time


def generate_naive_answer2(question: str, snippets: List[str]) -> str:
    """Generate a simple answer based on the top snippets"""
    context = " ".join(snippets[:2])  # Use top 2 snippets to form a response

    # Basic rule-based answering based on question keywords
    if "vacation" in question.lower():
        return "Based on company policy: Employees receive 15 vacation days per year."
    elif "remote" in question.lower() or "work from home" in question.lower():
        return "Remote work policy: Up to 3 days per week with manager approval."
    elif "health" in question.lower() or "insurance" in question.lower():
        return "Healthcare includes dental and vision coverage with $20 copays."
    else:
        return f"I found this information: {context[:150]}..."  # Provide a short snippet as a fallback
def generate_naive_answer(question: str, snippets: List[str]) -> str:
    """Generate a simple answer based on the top snippets"""
    context = " ".join(snippets[:2])  # Use top 2 snippets to form a response

    # Basic rule-based answering based on question keywords
    if "vacation" in question.lower():
        return "Based on company policy: Employees receive 15 vacation days per year."
    elif "remote" in question.lower() or "work from home" in question.lower():
        return "Remote work policy: Up to 3 days per week with manager approval."
    elif "health" in question.lower() or "insurance" in question.lower():
        return "Healthcare includes dental and vision coverage with $20 copays."
    elif "maternity" in question.lower() or "paternity" in question.lower():
        return "Sorry, I couldn't find specific information about maternity leave. Please consult HR."
    else:
        return f"I found this information: {context[:150]}..."

# Main endpoint for answering questions
@app.post("/answer", response_model=QueryResponse)
async def answer_question(request: QueryRequest):
    """Retrieve the top answers to a given question"""
    
    # Apply guardrail to block inappropriate topics
    if not check_guardrail(request.question):
        raise HTTPException(
            status_code=400, 
            detail="Query blocked: This topic requires HR consultation"
        )
    
    # Start timing the entire processing
    start_time = time.time()

    # Method 1: Cosine Similarity (preferred method for semantic search)
    cosine_snippets, cosine_time = search_cosine(request.question, request.top_k)

    # Method 2: Dot Product (faster but less accurate)
    dot_snippets, dot_time = search_dot_product(request.question, request.top_k)

    # Choose the best method based on cosine similarity results (generally better for semantic tasks)
    final_snippets = cosine_snippets
    method_used = "cosine_similarity"
    search_time = cosine_time

    # Generate an answer based on the top snippets
    answer = generate_naive_answer(request.question, final_snippets)

    total_time = time.time() - start_time  # Total processing time

    # Return the response with the question, answer, top snippets, method used, and processing time
    return QueryResponse(
        question=request.question,
        answer=answer,
        top_snippets=final_snippets,
        search_method=method_used,
        processing_time=total_time
    )


request_count = 0
guardrail_blocks = 0

@app.get("/metrics")
async def get_metrics():
    """Return metrics for monitoring purposes"""
    return {
        "total_requests": request_count,
        "guardrail_blocks": guardrail_blocks,
        "guardrail_block_rate": guardrail_blocks / max(request_count, 1),
        "average_latency": "N/A"  
        
    }


@app.get("/")
async def root():
    """Root endpoint message"""
    return {"message": "Retrieval QA Service - POST to /answer with your question"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
