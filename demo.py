import sys
import os
from dotenv import load_dotenv
import signal
from combinators_llm import CombinatorsLlm
from combinators_llm.utils import check_proof, check_proof_batch

load_dotenv()

BEAM_SIZE = 4
MAX_NUM_SIZE = 100
DEBUG = os.getenv("DEBUG") is not None


# Graceful exit on Ctrl+C
def handle_sigint(sig, frame):
    print("\n🛑  Exiting demo. Goodbye!")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)


def main():
    llm = CombinatorsLlm()

    print("✨ Interactive Combinators LLM Demo ✨")

    if DEBUG:
        print("🔍  Debug mode is ON")

    # Prompt for BEAM SIZE or Greedy
    print("Select generation method:")
    print("1. Greedy Decoding")
    print(f"2. Beam Search Decoding (beam_size={BEAM_SIZE})")
    method_choice = input("Enter your choice (1-2): ").strip()
    if method_choice not in {"1", "2"}:
        print("⚠️  Invalid choice. Defaulting to Greedy Decoding.\n")
        method_choice = "1"

    print("Type a logical type (e.g. `A -> B -> A`) and press Enter.")
    print("Press Ctrl+C to exit.\n")

    while True:
        try:
            type_str = input("🧩  Type: ").strip()
            if not type_str:
                print("⚠️  Please enter a non-empty type.\n")
                continue

            if method_choice == "1":
                print("🔮  Generating term with greedy decoding...")
                term = llm.generate(type_str)
                print(f"✅  Generated term: {term}\n")

                print("🔍  Checking proof validity...")
                is_valid = check_proof(type_str, term)

                if is_valid:
                    print("🎉  The generated term is a valid proof!\n")
                else:
                    print("❌  The generated term is NOT a valid proof.\n")

            elif method_choice == "2":
                print("🔮  Generating terms with beam search...")
                terms = llm.beam_generate(
                    type_str, beam_size=BEAM_SIZE, max_num_sequences=MAX_NUM_SIZE
                )

                print(f"✅  Generated {len(terms)} candidate term(s):\n")

                # Check all proofs in batch
                batch = [(type_str, term) for term in terms]
                results = check_proof_batch(batch)

                valid_terms = []
                for i, (term, is_valid) in enumerate(zip(terms, results), 1):
                    print(f"   {i}. {term}")

                    if is_valid:
                        valid_terms.append(term)
                        print(f"      ✓ Valid proof")
                    else:
                        print(f"      ✗ Invalid proof")

                print()
                if valid_terms:
                    print(
                        f"🎉  {len(valid_terms)}/{len(terms)} proof(s) verified successfully!\n"
                    )
                else:
                    print("❌  No valid proofs found among the candidates.\n")

            else:
                print("⚠️  Unknown generation method selected.\n")
                handle_sigint(None, None)
        except KeyboardInterrupt:
            # handled by signal, but just in case inside loop
            handle_sigint(None, None)
        except Exception as e:
            print(f"💥  Error: {e}\n")

            if DEBUG:
                raise e


if __name__ == "__main__":
    main()
