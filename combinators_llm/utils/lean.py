import tempfile
import subprocess


def check_proof(type: str, term: str) -> bool:

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(
            "universe u\nvariable {α β γ : Type u}\ndef s (f: α → β → γ) (g: α → β) (x: α) : γ := f x (g x)\ndef k (x: α) (_: β) : α := x\n"
        )
        tmp.write(
            "variable { " + " ".join([chr(65 + i) for i in range(26)]) + " : Type u }\n"
        )
        tmp.write(f"#check ({term.lower()} : {type})\n")

    try:

        result = subprocess.run(["lean", tmp_path], capture_output=True, text=True)

        if "error" in result.stdout.lower():
            return False
        else:
            return True

    finally:
        import os

        os.remove(tmp_path)
