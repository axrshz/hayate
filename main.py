from hayate.engine.engine import Engine


def main():
    engine = Engine("Qwen/Qwen3-4B")
    output = engine.generate_text("Explain Artificial General Intelligence")
    print(output.response)


if __name__ == "__main__":
    main()
