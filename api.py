import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from research import run_dual_agent  # Import your existing function
from fastapi.middleware.cors import CORSMiddleware  # Add this

app = FastAPI()

# Add these lines before your routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str] = []  # Optional: Add sources if your research agent collects them

@app.get("/")
async def root():
    return {"message": "Hello World"}
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Endpoint to submit questions to your dual-agent system"""
    answer = run_dual_agent(request.question)
    return {"answer": answer, "sources": []}  # Customize with actual sources if available

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)