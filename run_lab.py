from llm_scratch_lab.lab import GPTLab


def main():
    gpt_lab = GPTLab()
    print("Running GPT Lab...")

    gpt_lab.run("Hello, I am", with_manual_seed=True)

if __name__ == "__main__":
    main()