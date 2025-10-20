import tempfile
import subprocess
import json
from typing import Tuple, List
import os

LEAN_HEADER = """
import Lean
open Lean Elab Command Meta Term

syntax (name := checkStr) "#check_str" str str : command

@[command_elab checkStr]
def elabCheckStr : CommandElab := fun stx => do
  try
    -- Extract the two literal strings
    let some tyStr   := stx[1].isStrLit? | throwError "expected first argument to be a string literal"
    let some termStr := stx[2].isStrLit? | throwError "expected second argument to be a string literal"

    -- Parse the strings into Syntax; runParserCategory returns Except
    let tyStx ← match Parser.runParserCategory (← getEnv) `term tyStr with
      | .ok stx  => pure stx
      | .error e => throwError "parse type failed: {e}"

    let termStx ← match Parser.runParserCategory (← getEnv) `term termStr with
      | .ok stx  => pure stx
      | .error e => throwError "parse term failed: {e}"

    -- Elaborate inside TermElabM
    let ok ← liftTermElabM do
      let tyExpr   ← elabType tyStx
      let termExpr ← elabTerm termStx (some tyExpr)
      let gotTy ← inferType termExpr
      isDefEq gotTy tyExpr

    if ok then
      logInfo m!"✅ {termStr} : {tyStr}"
    else
      logInfo m!"❌ {termStr} : {tyStr} (type mismatch)"
  catch err =>
    -- Catch any error and print ❌ instead of failing
    let termStr := stx[2].isStrLit?.getD "<unknown term>"
    let tyStr   := stx[1].isStrLit?.getD "<unknown type>"
    logInfo m!"❌ {termStr} : {tyStr} — {← err.toMessageData.toString}"
  pure ()

universe u
variable {α β γ : Type u}
def s (f: α → β → γ) (g: α → β) (x: α) : γ := f x (g x)
def k (x: α) (_: β) : α := x
variable { A B C D E F G H I J L M N O P Q R T U V W X Y Z : Type u }
"""


def check_proof(type: str, term: str) -> bool:

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(LEAN_HEADER)
        tmp.write(f'#check_str "{type}"  "{term.lower()}"\n')

    try:

        result = subprocess.run(
            ["lean", "--json", tmp.name],
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = result.stdout + "\n" + result.stderr
        for line in output.splitlines():

            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
                if msg.get("severity") == "information":
                    text = msg.get("data").strip()
                    if "✅" in text:
                        return True
                    elif "❌" in text:
                        return False
            except json.JSONDecodeError:
                continue
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
            tmp.write(f'#check_str "{pair[0]}"  "{pair[1].lower()}" \n')
    try:

        results = subprocess.run(
            ["lean", "--json", tmp.name],
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = results.stdout + "\n" + results.stderr

        val: List[bool] = []
        for line in output.splitlines():

            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
                if msg.get("severity") == "information":

                    text = msg["data"].strip()

                    if text.startswith("✅"):
                        val.append(True)
                    elif text.startswith("❌"):
                        val.append(False)
            except json.JSONDecodeError:
                continue

    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    assert len(batch) == len(val)

    return val
