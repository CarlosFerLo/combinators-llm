# Combinators LLM

A basic 2017 transformer trained with the (combinators-dataset)[https://githgub.com/carlosferlo/combinators-dataset].
The main goal is to navigate the probability space of proofs by using this model.

To use this model clone this repo and run:

```
from combinators_llm import

model = CombinatorsLlm()

model("A -> B -> A") # Run the model with a single string
model(["A -> B -> A", "A -> A]) # Run the model in batches

```
