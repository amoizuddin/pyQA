import torch
print(torch.backends.mps.is_available())  # True means GPU is usable
print(torch.backends.mps.is_built())      # Should also be True

from transformers import pipeline
import torch

device = 0 if torch.backends.mps.is_available() else -1
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=device)

context = "hugging face is a company that develops tools for machine learning"
question = "what does hugging face develop?"

print(qa_pipeline(question=question, context=context))
