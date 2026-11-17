
import torch

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    device
):
    model.eval()

    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens
    )

    return tokenizer.decode(output_ids[0].tolist())
