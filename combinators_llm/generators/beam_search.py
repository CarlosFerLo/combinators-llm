import torch
from typing import Optional
from ..utils import causal_mask


def beam_search_decode(
    model,
    source,
    source_mask,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    beam_size=4,
    length_penalty=1.0,
    max_num_sequences: Optional[int] = None,
):
    """
    Beam search decoder with length normalization.

    Args:
        model: The transformer model
        source: Source sequence tensor (1, seq_len)
        source_mask: Source mask tensor
        tokenizer_src: Source tokenizer
        tokenizer_tgt: Target tokenizer
        max_len: Maximum generation length
        device: Device to run on
        beam_size: Number of beams to maintain
        length_penalty: Length penalty coefficient (alpha). Higher values favor longer sequences.
                       Score = log_prob / (length ** length_penalty)
                       Default 1.0 provides balanced normalization
        max_num_sequences: If set, limits the number of returned sequences to this value.
                           Otherwise, returns all beams found.

    Returns:
        List of tuples (sequence, normalized_score) for the best beams, sorted by score (best first)
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every token
    encoder_output = model.encode(source, source_mask)

    # Initialize beams with SOS token
    # Each beam is a tuple: (sequence, log_probability)
    active_beams = [
        (
            torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device),
            0.0,  # Initial log probability
        )
    ]

    # Store finished beams separately
    finished_beams = []

    for step in range(max_len - 1):
        all_candidates = []

        # Expand each active beam
        for beam_seq, beam_score in active_beams:
            # Build the mask for the target (decoder input)
            decoder_mask = causal_mask(beam_seq.size(1)).type_as(source_mask).to(device)

            # Calculate the output of the decoder
            out = model.decode(encoder_output, source_mask, beam_seq, decoder_mask)

            # Get the probability distribution for the next token
            prob = model.project(out[:, -1])
            log_prob = torch.log_softmax(prob, dim=-1)

            # Get top beam_size tokens for this beam
            top_log_probs, top_indices = torch.topk(log_prob, beam_size, dim=-1)

            for i in range(beam_size):
                next_token = top_indices[0, i].item()
                token_log_prob = top_log_probs[0, i].item()

                # Create new sequence
                new_seq = torch.cat(
                    [
                        beam_seq,
                        torch.empty(1, 1).type_as(source).fill_(next_token).to(device),
                    ],
                    dim=1,
                )

                # Update log probability
                new_score = beam_score + token_log_prob

                # Check if this beam is finished
                is_finished = next_token == eos_idx

                all_candidates.append((new_seq, new_score, is_finished))

        # Separate finished and active candidates
        new_finished = []
        new_active = []

        for seq, score, is_finished in all_candidates:
            if is_finished:
                new_finished.append((seq, score))
            else:
                new_active.append((seq, score))

        # Add newly finished beams to the finished list
        finished_beams.extend(new_finished)

        # If no active beams remain, we're done
        if len(new_active) == 0:
            break

        # Apply length normalization ONLY for selecting top active beams
        # This helps compare beams of different lengths fairly during generation
        normalized_active = []
        for seq, score in new_active:
            seq_length = seq.size(1)
            normalized_score = score / (seq_length**length_penalty)
            normalized_active.append((seq, score, normalized_score))

        # Sort active beams by normalized score and keep top beam_size
        normalized_active.sort(key=lambda x: x[2], reverse=True)

        # Keep only the top beam_size active beams for the next iteration
        active_beams = [(seq, score) for seq, score, _ in normalized_active[:beam_size]]

        # Continue until max_len or all beams finish naturally
        # No early stopping - let all beams explore fully

    # Separate beams that finished naturally (with EOS) from those that reached max_len
    beams_with_eos = []
    beams_at_maxlen = []

    # Check finished beams for EOS token
    for seq, score in finished_beams:
        last_token = seq[0, -1].item()
        if last_token == eos_idx:
            beams_with_eos.append((seq, score))
        else:
            beams_at_maxlen.append((seq, score))

    # Add any remaining active beams that didn't finish to beams_at_maxlen
    # These are sequences that reached max_len
    for seq, score in active_beams:
        beams_at_maxlen.append((seq, score))

    # Apply final normalization to all beams for fair comparison
    normalized_with_eos = []
    for seq, score in beams_with_eos:
        seq_length = seq.size(1)
        normalized_score = score / (seq_length**length_penalty)
        normalized_with_eos.append((seq.squeeze(0), normalized_score, True))

    normalized_at_maxlen = []
    for seq, score in beams_at_maxlen:
        seq_length = seq.size(1)
        normalized_score = score / (seq_length**length_penalty)
        normalized_at_maxlen.append((seq.squeeze(0), normalized_score, False))

    # Combine all beams, prioritizing those with EOS
    # Beams with EOS are generally better as they ended naturally
    all_final_beams = normalized_with_eos + normalized_at_maxlen

    # Sort by normalized score (best first)
    all_final_beams.sort(key=lambda x: x[1], reverse=True)

    # Return only top max_num_sequences results if specified, otherwise return all beams
    if max_num_sequences is not None:
        results = [
            (seq, score) for seq, score, _ in all_final_beams[:max_num_sequences]
        ]
    else:
        results = [(seq, score) for seq, score, _ in all_final_beams]

    return results
