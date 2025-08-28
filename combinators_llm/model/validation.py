import torch

from combinators_llm.generators import greedy_decode
from combinators_llm.utils.lean import check_proof

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, global_step, writer) -> None :
    model.eval()
    
    count = 0
    
    with torch.no_grad() :
        for batch in validation_ds :
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be one for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            type_text = batch["type_text"][0]
            term_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            if check_proof(type_text, term_text) :
                count += 1
                
    writer.add_scalar("validation_acc", count / len(validation_ds), global_step)