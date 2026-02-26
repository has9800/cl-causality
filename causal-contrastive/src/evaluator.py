"""Trace generation and Nemotron scoring."""

import json
import os

import torch
from openai import OpenAI


FEW_SHOT_PREFIX = """Prompt: Why does ice melt when salt is applied?
Trace: Cause: salt dissolves into the liquid film on ice. Mechanism: dissolved ions lower the freezing point through colligative effects. Effect: ice melts at temperatures below 0C.

Prompt: Why does exercise reduce blood pressure?
Trace: Cause: regular aerobic exercise is performed. Mechanism: repeated cardiac demand improves arterial elasticity and reduces peripheral resistance. Effect: resting blood pressure decreases over weeks.

"""


def generate_traces(model, cfg) -> list[dict]:
    """Generate traces for all evaluation prompts."""
    eval_cfg = cfg["evaluation"]
    results = []

    prompts = []
    with open(cfg["data"]["causal_prompts"], encoding="utf-8") as file:
        for line in file:
            prompts.append(json.loads(line)["prompt"])

    prompts = prompts[: eval_cfg["num_prompts"]]
    model.model.eval()

    for prompt in prompts:
        full_prompt = FEW_SHOT_PREFIX + f"Prompt: {prompt}\nTrace:"
        inputs = model.tokenize([full_prompt], max_length=1024)
        prefix_len = inputs["input_ids"].shape[1]

        traces = []
        for _ in range(eval_cfg["traces_per_prompt"]):
            with torch.no_grad():
                outputs = model.model.generate(
                    **inputs,
                    max_new_tokens=eval_cfg["max_new_tokens"],
                    do_sample=True,
                    temperature=eval_cfg["temperature"],
                    pad_token_id=model.tokenizer.eos_token_id,
                )
            generated_ids = outputs[0][prefix_len:]
            completion = model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            for stop in ["Prompt:", "\n\n", "\nA:", "\nQ:", "\nQuestion:", "\nAnswer:"]:
                if stop in completion:
                    completion = completion[: completion.index(stop)].strip()

            traces.append(completion)

        results.append({"prompt": prompt, "traces": traces})

    return results


def _extract_nemotron_score(response) -> float:
    """Extract reward score from potential API response shapes."""
    choice = response.choices[0]

    if getattr(choice, "message", None) and getattr(choice.message, "content", None):
        content = choice.message.content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    try:
                        return float(part["text"])
                    except (TypeError, ValueError):
                        continue
        else:
            try:
                return float(content)
            except (TypeError, ValueError):
                pass

    for attr in ["reward", "score"]:
        if hasattr(choice, attr):
            try:
                return float(getattr(choice, attr))
            except (TypeError, ValueError):
                continue

    raise ValueError(f"Could not parse Nemotron score from response: {response}")


def score_with_nemotron(traces_data: list[dict], cfg: dict) -> list[dict]:
    """
    Score each trace using Nemotron reward model via API.
    Returns traces_data with 'scores' added to each entry.
    """
    nem_cfg = cfg["nemotron"]
    client = OpenAI(
        base_url=nem_cfg["base_url"],
        api_key=os.environ[nem_cfg["api_key_env"]],
    )

    scored_data = []
    for entry in traces_data:
        prompt = entry["prompt"]
        scores = []
        for trace in entry["traces"]:
            try:
                response = client.chat.completions.create(
                    model=nem_cfg["model"],
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": trace},
                    ],
                )
                score = _extract_nemotron_score(response)
            except Exception as error:
                print(f"Nemotron scoring error: {error}")
                score = 0.0
            scores.append(score)

        scored_data.append(
            {
                "prompt": prompt,
                "traces": entry["traces"],
                "scores": scores,
                "mean_score": sum(scores) / len(scores),
                "best_score": max(scores),
                "best_trace": entry["traces"][scores.index(max(scores))],
            }
        )

    return scored_data
