from hayate.engine.engine import Engine

engine = Engine("Qwen/Qwen3-4B")
print(engine.generate_text("Explain AGI"))

outputs = engine.generate_text([
    "Explain AGI",
    "What is vLLM?",
    "Tell me about SGLang"
])

for o in outputs:
    print(o)
