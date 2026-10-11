"""Shared helpers: Bedrock client construction and bootstrap confidence intervals.

Credentials are resolved by the standard AWS credential chain (e.g. `aws configure`,
an AWS profile, or an instance role); this code never reads secrets itself.
"""
import os
from typing import Optional


def make_client(backend: str = "bedrock", region: Optional[str] = None):
    """Anthropic-SDK client for Claude on Amazon Bedrock (default) or the first-party API."""
    if backend == "bedrock":
        from anthropic import AnthropicBedrock
        region = region or os.environ.get("AWS_REGION") or "us-west-2"
        return AnthropicBedrock(aws_region=region, max_retries=4)
    import anthropic
    return anthropic.Anthropic()


def resolve_model(model: str, backend: str, profile: str = "us") -> str:
    """Bedrock needs a cross-region inference-profile ID, e.g. 'us.anthropic.claude-opus-5'."""
    if backend != "bedrock":
        return model
    return f"{profile}.anthropic.{model.split('.')[-1]}"


def bootstrap_ci(values, B=2000, seed=0):
    """Mean and 95% percentile-bootstrap CI over cases, as [mean%, lo%, hi%]."""
    import numpy as np
    v = np.asarray(values, float)
    if len(v) == 0:
        return [0.0, 0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return [round(float(100 * v.mean()), 1),
            round(float(100 * np.percentile(means, 2.5)), 1),
            round(float(100 * np.percentile(means, 97.5)), 1)]
