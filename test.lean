import Lean
open Lean Meta Elab Command

elab "#check_str" ty:str term:str : command => do
  IO.println s!"running!"
  try
    let tyStx ← liftCoreM <| Parser.runParserCategory (← getEnv) `term ty.getString
    let termStx ← liftCoreM <| Parser.runParserCategory (← getEnv) `term term.getString
    let tyExpr ← elabTerm tyStx none
    let termExpr ← elabTerm termStx (some tyExpr)
    let ok ← isDefEq (← inferType termExpr) tyExpr
    if ok then
      IO.println s!"✅ {term.getString} : {ty.getString}"
    else
      IO.println s!"❌ {term.getString} : {ty.getString}"
  catch e =>
    IO.println s!"❌ {term.getString} : {ty.getString} — {← e.toMessageData.toString}"

#check_str "A → B → A" "fun x y => x"
