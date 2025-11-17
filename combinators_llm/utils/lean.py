import tempfile
import subprocess
import json
from typing import Tuple, List, Optional
import os
import logging
import re

logger = logging.getLogger(__name__)

# Markers for proof validation results
SUCCESS_MARKER = "✅"
FAILURE_MARKER = "❌"

LEAN_HEADER = """
import Lean
open Lean Elab Command Meta Term

universe u
variable {α β γ : Type u}
def s (f: α → β → γ) (g: α → β) (x: α) : γ := f x (g x)
def k (x: α) (_: β) : α := x

-- Declare type variables as axioms so they're in scope everywhere
axiom A : Type u
axiom B : Type u
axiom C : Type u
axiom D : Type u
axiom E : Type u
axiom F : Type u
axiom G : Type u
axiom H : Type u
axiom I : Type u
axiom J : Type u
axiom L : Type u
axiom M : Type u
axiom N : Type u
axiom O : Type u
axiom P : Type u
axiom Q : Type u
axiom R : Type u
axiom T : Type u
axiom U : Type u
axiom V : Type u
axiom W : Type u
axiom X : Type u
axiom Y : Type u
axiom Z : Type u

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
      let expectedTy ← elabType tyStx
      -- Elaborate the term WITHOUT an expected type to get its true inferred type
      let termExpr ← elabTerm termStx none
      let actualTy ← inferType termExpr
      -- Check if the types are definitionally equal
      isDefEq actualTy expectedTy

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

"""


def check_proof(type: str, term: str, timeout: int = 60) -> bool:
    """Check if a term is a valid proof for a given type using Lean.

    Args:
        type: The type to check (e.g., "A -> B -> A")
        term: The term to validate (e.g., "k")
        timeout: Maximum time in seconds for Lean execution (default: 60)

    Returns:
        True if the term is a valid proof, False otherwise
    """
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(LEAN_HEADER)
            tmp.write(f'#check_str "{type}"  "{term.lower()}"\n')

        logger.debug(f"Running Lean on temporary file: {tmp_path}")

        result = subprocess.run(
            ["lean", "--json", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            logger.warning(f"Lean exited with non-zero code: {result.returncode}")

        output = result.stdout + "\n" + result.stderr
        logger.debug(f"Lean output:\n{output}")

        # Parse JSON output line by line
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
                if msg.get("severity") == "information":
                    data = msg.get("data", "")
                    if not data:
                        continue

                    text = data.strip()

                    # More robust parsing: check if line starts with markers
                    if text.startswith(SUCCESS_MARKER):
                        logger.debug(f"Proof validated successfully: {type} :: {term}")
                        return True
                    elif text.startswith(FAILURE_MARKER):
                        logger.debug(f"Proof validation failed: {type} :: {term}")
                        return False

            except json.JSONDecodeError as e:
                logger.debug(f"Skipping non-JSON line: {line[:50]}...")
                continue

        logger.warning(
            f"No validation result found in Lean output for: {type} :: {term}"
        )
        return False

    except subprocess.TimeoutExpired:
        logger.error(f"Lean execution timed out after {timeout}s for: {type} :: {term}")
        return False
    except FileNotFoundError:
        logger.error(
            "Lean executable not found. Make sure Lean is installed and in PATH."
        )
        return False
    except Exception as e:
        logger.error(f"Unexpected error during proof checking: {e}")
        return False
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {tmp_path}: {e}")


def check_proof_batch(batch: List[Tuple[str, str]], timeout: int = 60) -> List[bool]:
    """Check multiple proofs in a single Lean invocation for better performance.

    Args:
        batch: List of (type, term) tuples to validate
        timeout: Maximum time in seconds for Lean execution (default: 60)

    Returns:
        List of boolean values indicating validity for each proof in order
    """
    if not batch:
        logger.warning("Empty batch provided to check_proof_batch")
        return []

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".lean", delete=False
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(LEAN_HEADER)

            for pair in batch:
                tmp.write(f'#check_str "{pair[0]}"  "{pair[1].lower()}" \n')

        logger.debug(f"Checking batch of {len(batch)} proofs using Lean")
        logger.debug(f"Temporary file: {tmp_path}")

        results = subprocess.run(
            ["lean", "--json", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if results.returncode != 0:
            logger.warning(f"Lean exited with non-zero code: {results.returncode}")

        output = results.stdout + "\n" + results.stderr
        logger.debug(f"Lean output:\n{output}")

        val: List[bool] = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
                if msg.get("severity") == "information":
                    data = msg.get("data", "")
                    if not data:
                        continue

                    text = data.strip()

                    # More robust parsing with explicit marker checking
                    if text.startswith(SUCCESS_MARKER):
                        val.append(True)
                        logger.debug(f"Proof {len(val)}/{len(batch)}: Valid")
                    elif text.startswith(FAILURE_MARKER):
                        val.append(False)
                        logger.debug(f"Proof {len(val)}/{len(batch)}: Invalid")

            except json.JSONDecodeError as e:
                logger.debug(f"Skipping non-JSON line: {line[:50]}...")
                continue
            except KeyError as e:
                logger.debug(f"Skipping malformed JSON message: {e}")
                continue

        # Validate that we got results for all proofs
        if len(val) != len(batch):
            logger.error(
                f"Result mismatch: Expected {len(batch)} results, got {len(val)}. "
                f"This may indicate Lean compilation errors or timeout issues."
            )

            # Fill missing results with False to maintain alignment
            missing_count = len(batch) - len(val)
            if missing_count > 0:
                logger.warning(f"Filling {missing_count} missing results with False")
                val.extend([False] * missing_count)
            elif missing_count < 0:
                # Shouldn't happen, but handle it anyway
                logger.error(
                    f"Got more results than expected, truncating to {len(batch)}"
                )
                val = val[: len(batch)]

        logger.info(f"Batch validation complete: {sum(val)}/{len(batch)} proofs valid")
        return val

    except subprocess.TimeoutExpired:
        logger.error(
            f"Lean execution timed out after {timeout}s for batch of {len(batch)} proofs"
        )
        # Return all False on timeout
        return [False] * len(batch)
    except FileNotFoundError:
        logger.error(
            "Lean executable not found. Make sure Lean is installed and in PATH."
        )
        return [False] * len(batch)
    except Exception as e:
        logger.error(f"Unexpected error during batch proof checking: {e}")
        return [False] * len(batch)
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {tmp_path}: {e}")


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    print("=" * 60)
    print("Testing Lean Proof Checker")
    print("=" * 60)

    # Test batch with known results
    batch = [
        ("A -> A", "s k k"),  # True - identity combinator
        ("A -> B -> A", "k"),  # True - K combinator
        ("A -> B", "k"),  # False - type mismatch
        ("A -> B", "s ( k ( k k ) ) k ( s k ( k k ) ) k "),  # False - type mismatch
        ("A -> A -> A", "k k"),  # False - partial application
    ]

    print(f"\nTesting batch of {len(batch)} proofs...")
    print("-" * 60)

    result = check_proof_batch(batch)

    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)

    valid_count = sum(result)
    for i, (pair, res) in enumerate(zip(batch, result), 1):
        status = "✅ VALID" if res else "❌ INVALID"
        print(f"{i}. {status}")
        print(f"   Type: {pair[0]}")
        print(f"   Term: {pair[1]}")
        print()

    print("=" * 60)
    print(f"Total: {valid_count}/{len(batch)} proofs validated successfully")
    print("=" * 60)
