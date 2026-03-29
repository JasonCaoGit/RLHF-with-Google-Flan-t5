# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
messages = [
    {"role": "user", "content": "Write a python function that prints the fiboncci sequence!"},
]
response = pipe(messages)
print(response[0]['generated_text'])