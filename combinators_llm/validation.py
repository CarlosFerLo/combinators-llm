import torch

import logging
from tqdm import tqdm

from typing import List, Tuple

from .generators import greedy_decode_batch
from .utils.lean import check_proof_batch


def run_validation(
    val_name,
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    global_step,
    run,
) -> float:
    model.eval()
    logging.getLogger(__name__)

    count = 0
    total = 0

    with torch.no_grad():
        batch_iterator = tqdm(validation_ds, desc=f"Running {val_name}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            model_out = greedy_decode_batch(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )
            model_out = model_out.detach().cpu().numpy()

            pairs: List[Tuple[str, str]] = []

            for i in range(encoder_input.size(0)):
                type_text: str = batch["type_text"][i]

                term_text = tokenizer_tgt.decode(model_out[i])
                term_text: str = term_text.replace("[SOS]", "").split("[EOS]")[0]

                pairs.append((type_text, term_text))

            total += len(batch["type_text"])
            res = check_proof_batch(pairs)
            count += sum(res)

            batch_iterator.set_postfix({"acc": f"{count / total :6.3f}"})

    results = count / total

    run.log({f"{val_name}_acc": results}, global_step)
    logging.info(f"Validation accuracy: {results}")

    return results
