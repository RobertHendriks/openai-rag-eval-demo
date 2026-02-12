"""
Prompt Registry — Externalizes and versions system prompts.

For this demo, prompts are stored as YAML files
in the prompts/ directory with version metadata. 
Store in a database or leverage LangSmtih/Humanloop
to harden for production.

Usage:
    from src.prompt_registry import get_prompt
    prompt = get_prompt("customer_support")           # latest version
    prompt = get_prompt("customer_support", "1.0")    # specific version
"""

import os
import json
import yaml
from datetime import datetime

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def list_prompts():
    """List all available prompts and their versions."""
    prompts = {}
    if not os.path.exists(PROMPTS_DIR):
        return prompts

    for filename in os.listdir(PROMPTS_DIR):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            filepath = os.path.join(PROMPTS_DIR, filename)
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            name = data.get("name", filename)
            if name not in prompts:
                prompts[name] = []
            prompts[name].append({
                "version": data.get("version", "unknown"),
                "description": data.get("description", ""),
                "file": filename
            })

    return prompts


def get_prompt(name, version=None):
    """
    Retrieve a prompt by name and optional version.

    Args:
        name: The prompt name (e.g., "customer_support")
        version: Optional version string (e.g., "1.0"). If None, returns latest.

    Returns:
        dict with keys: name, version, system_prompt, description, metadata
    """
    if not os.path.exists(PROMPTS_DIR):
        raise FileNotFoundError(f"Prompts directory not found: {PROMPTS_DIR}")

    candidates = []

    for filename in os.listdir(PROMPTS_DIR):
        if not (filename.endswith(".yaml") or filename.endswith(".yml")):
            continue

        filepath = os.path.join(PROMPTS_DIR, filename)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        if data.get("name") == name:
            candidates.append(data)

    if not candidates:
        raise ValueError(f"No prompt found with name: {name}")

    if version:
        match = [c for c in candidates if str(c.get("version")) == str(version)]
        if not match:
            available = [str(c.get("version")) for c in candidates]
            raise ValueError(
                f"Version {version} not found for '{name}'. "
                f"Available: {available}"
            )
        return match[0]

    # Return latest version (highest version string)
    candidates.sort(key=lambda x: str(x.get("version", "0")), reverse=True)
    return candidates[0]


if __name__ == "__main__":
    print("Available prompts:")
    for name, versions in list_prompts().items():
        for v in versions:
            print(f"  {name} v{v['version']}: {v['description']}")

    print("\nLoading latest customer_support prompt:")
    prompt = get_prompt("customer_support")
    print(f"  Version: {prompt['version']}")
    print(f"  System prompt: {prompt['system_prompt'][:100]}...")