from typing import Tuple, Union

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from nlp.config import settings


def load_model_and_tokenizer(
        model_path: str, device: Union[str, torch.device], cache_dir: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a pre-trained causal language model and its tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    return model, tokenizer


def contrastive_generation(
        amateur_model: AutoModelForCausalLM,
        expert_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_tokens: int = 500,
        alpha: float = 0.1,
        temperature: float = 1.0,
        device: Union[str, torch.device] = "cuda",
) -> str:
    """
    Generate text using contrastive decoding between an amateur and an expert model.
    """
    # Tokenize input prompt
    input_tokens: Tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_tokens: Tensor = input_tokens

    for _ in tqdm(range(max_tokens), desc="Generating text"):
        with torch.no_grad():
            amateur_logits: Tensor = (
                    amateur_model(generated_tokens).logits[:, -1, :].to(device) / temperature
            )
            expert_logits: Tensor = (
                    expert_model(generated_tokens).logits[:, -1, :].to(device) / temperature
            )

            contrastive_logits: Tensor = expert_logits - alpha * amateur_logits
            next_token: Tensor = torch.argmax(contrastive_logits, dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


user_message = """Give a very very brief docstring for the following function:\n```
function updateEloScores(
    scores,
    results,
    kFactor = 4,
) {
    for (const result of results) {
            const { first, second, outcome } = result;
            const firstScore = scores[first] ?? 1000;
            const secondScore = scores[second] ?? 1000;

            const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
            const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
            let sa = 0.5;
            if (outcome === 1) {
                    sa = 1;
            } else if (outcome === -1) {
                    sa = 0;
            }
            scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
            scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
    }
    return scores;
}\n```"""


def main() -> None:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    expert_path: str = "Qwen/Qwen2.5-3B-Instruct"
    amateur_path: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    # Load models and tokenizers
    expert_model, _ = load_model_and_tokenizer(expert_path, device, settings.HUGGINGFACE_CACHE_DIR)
    amateur_model, amateur_tokenizer = load_model_and_tokenizer(
        amateur_path, device, settings.HUGGINGFACE_CACHE_DIR
    )

    # Prepare prompt
    prompt: str = amateur_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_message},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    response: str = contrastive_generation(
        amateur_model,
        expert_model,
        amateur_tokenizer,
        prompt,
        device=device,
    )
    print(response)


if __name__ == "__main__":
    main()
