from hayate.engine.engine import Engine

engine = Engine("Qwen/Qwen3-4B")

output = engine.generate_text("Explain Artificial General Intelligence")

print(output.response)
