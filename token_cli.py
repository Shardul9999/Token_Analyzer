import tiktoken
from dataclasses import dataclass

PRICING = {
    "gpt-4o": {
        "input": 5.00,
        "output": 15.00,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
}

MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
}

RESPONSE_SIZES = {
    "short": 100,
    "medium": 500,
    "long": 2000,
}

class ContextWindowWarning(Exception):
    pass


@dataclass
class TokenUsage:
    input_tokens: int
    estimated_output_tokens: int
    model: str
    estimated_cost_usd: float
    context_utilization_pct: float

    def warn_if_near_limit(self, threshold: float = 0.85):

        if self.context_utilization_pct > threshold:
            raise ContextWindowWarning(
                f" Using {self.context_utilization_pct:.0%} "
                f"of context window"
            )


def estimate_tokens(
    prompt: str,
    model: str,
    response_size: str
) -> TokenUsage:

    enc = tiktoken.encoding_for_model(model)

    input_tokens = len(enc.encode(prompt))

    estimated_output_tokens = RESPONSE_SIZES[response_size]

    pricing = PRICING[model]

    input_cost = (
        input_tokens / 1_000_000
    ) * pricing["input"]

    output_cost = (
        estimated_output_tokens / 1_000_000
    ) * pricing["output"]

    total_cost = input_cost + output_cost

    ctx_limit = MODEL_CONTEXT_LIMITS.get(
        model,
        128_000
    )

    context_usage = (
        input_tokens + estimated_output_tokens
    ) / ctx_limit

    return TokenUsage(
        input_tokens=input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        model=model,
        estimated_cost_usd=round(total_cost, 8),
        context_utilization_pct=context_usage
    )

# Main Function -->
def main():    
    print("Available Models:")

    for model in PRICING.keys():
        print(f" - {model}")

    prompt = input("\nEnter your prompt:\n> ")
    
    model = input(
        "\nChoose model:\n"
        "(gpt-4o / gpt-4o-mini)\n> "
    ).strip()

    if model not in PRICING:
        print("\n Invalid model selected")
        return

    print("\nChoose expected response size:")
    print("1. Short")
    print("2. Medium")
    print("3. Long")

    size_choice = input("\nEnter choice (1/2/3):\n> ").strip()

    size_map = {
        "1": "short",
        "2": "medium",
        "3": "long",
    }

    if size_choice not in size_map:
        print("\n Invalid choice")
        return

    response_size = size_map[size_choice]

    usage = estimate_tokens(
        prompt=prompt,
        model=model,
        response_size=response_size
    )

    print(f"Model: {usage.model}")

    print(f"Input Tokens: {usage.input_tokens}")

    print(
        f"Estimated Output Tokens: "
        f"{usage.estimated_output_tokens}"
    )

    print(
        f"Estimated Cost: "
        f"${usage.estimated_cost_usd}"
    )

    print(
        f"Context Usage: "
        f"{usage.context_utilization_pct:.2%}"
    )

    # Warning
    try:
        usage.warn_if_near_limit()

    except ContextWindowWarning as e:
        print(f"\n{e}")


if __name__ == "__main__":
    main()
