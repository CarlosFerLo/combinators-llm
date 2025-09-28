import tempfile
import subprocess
import json
from typing import Tuple, List, Dict, Union
import os

LEAN_HEADER = """universe u
variable {α β γ : Type u}
def s (f: α → β → γ) (g: α → β) (x: α) : γ := f x (g x)
def k (x: α) (_: β) : α := x

variable { A B C D E F G H I J L M N O P Q R T U V W X Y Z : Type u }
"""


def check_proof(type: str, term: str) -> bool:

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(LEAN_HEADER)
        tmp.write(f"#check ({term.lower()} : {type})\n")

    try:

        result = subprocess.run(
            ["lean", "--json", tmp.name], capture_output=True, text=True
        )

        if result.stdout.strip():
            out = json.loads(result.stdout)

            if out["severity"] == "error":
                return False
            else:
                return True
        return False

    finally:
        import os

        os.remove(tmp_path)


def check_proof_batch(batch: List[Tuple[str, str]]) -> List[bool]:

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".lean", delete=False
    ) as tmp:
        tmp.write(LEAN_HEADER)

        for pair in batch:
            tmp.write(f"#check ( {pair[1].lower()} : {pair[0]} )\n")
    try:

        results = subprocess.run(
            ["lean", "--json", tmp.name], capture_output=True, text=True
        )

        output: List[Dict[str, Union[str, int]]] = []
        for line in results.stdout.splitlines():

            if line.strip():
                out = json.loads(line)
                output.append(out)

        val: List[bool] = [out["severity"] != "error" for out in output]

    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    return val
