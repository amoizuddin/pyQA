from transformers import pipeline
import torch

class QAModel:
    def __init__(self):
        device = 0 if torch.backends.mps.is_available() else -1
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=device)

    def eval(self, question: str, context: str):
        return self.qa_pipeline(question=question, context=context)
    
model = QAModel()
