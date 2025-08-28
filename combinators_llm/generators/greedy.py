import torch
from combinators_llm.dataset import causal_mask

def greedy_decode (model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device) :
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    
    # Precompute the encoder output and reuse it for every token
    encoder_output = model.encode(source, source_mask)
    
    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True :
        if decoder_input.size(1) == max_len :
            break
        
        # Build the mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Get the next token
        prob = model.project(out[:, -1])
        
        # Select the token with the max probability (greedy search)
        _, next_word = torch.max(prob, dim=1)
        
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)
        
        if next_word == eos_idx :
            break
        
    return decoder_input.squeeze(0)