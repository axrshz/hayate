from hayate.engine.engine import Engine

engine = Engine("Qwen/Qwen3-4B")

outputs = engine.generate_text([
    "Explain Artificial General Intelligence"
])

for o in outputs:
    print(o.response)
