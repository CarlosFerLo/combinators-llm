import sys
import signal
from combinators_llm import CombinatorsLlm
from combinators_llm.utils import check_proof


# Graceful exit on Ctrl+C
def handle_sigint(sig, frame):
    print("\n🛑  Exiting demo. Goodbye!")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)


def main():
    llm = CombinatorsLlm()

    print("✨ Interactive Combinators LLM Demo ✨")
    print("Type a logical type (e.g. `A -> B -> A`) and press Enter.")
    print("Press Ctrl+C to exit.\n")

    while True:
        try:
            type_str = input("🧩  Type: ").strip()
            if not type_str:
                print("⚠️  Please enter a non-empty type.\n")
                continue

            print("🔮  Generating term...")
            term = llm.generate(type_str)
            print(f"✅  Generated term:\n   {term}\n")

            print("🧠  Checking proof...")
            ok = check_proof(type_str, term)
            if ok:
                print("🎉  Proof verified successfully!\n")
            else:
                print("❌  Proof failed. The term does not inhabit the given type.\n")

        except KeyboardInterrupt:
            # handled by signal, but just in case inside loop
            handle_sigint(None, None)
        except Exception as e:
            print(f"💥  Error: {e}\n")


if __name__ == "__main__":
    main()
