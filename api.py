from fastapi import FastAPI
from pydantic import BaseModel
from model import model

app = FastAPI()

class Question_Request(BaseModel):
    question: str
    context: str
@app.get("/")
def is_alive():
    return {"message":"I LIVE!"}
@app.post("/ask")
def ask_question(req: Question_Request):
    answer = model.eval(req.question, req.context)
    return {"question": req.question, "answer": answer}
