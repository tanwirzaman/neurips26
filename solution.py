"""Zero-compute entry: predict robust for every case; no trained model."""


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Return ordered native booleans for the Codabench interface."""
    return [True for _ in problems]
