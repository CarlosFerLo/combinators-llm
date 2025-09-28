import torch
from ..utils import causal_mask


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every token
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build the mask for the target (decoder input)
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])

        # Select the token with the max probability (greedy search)
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def greedy_decode_batch(model, source, source_mask, tokenizer_tgt, max_len, device):
    batch_size = source.size(0)
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)

    # Initialize decoder input with SOS token for all sequences in the batch
    decoder_input = torch.full(
        (batch_size, 1), sos_idx, dtype=source.dtype, device=device
    )

    # Track which sequences have finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        # Append next token
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)

        # Mark finished sequences
        finished |= next_word == eos_idx

        # Stop early if all sequences are finished
        if finished.all():
            break

    return decoder_input
