import json
import random
from pathlib import Path
from typing import Any, Literal

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser

from self_preservation.templates.roles import ROLES
from self_preservation.templates.user_prompt import (
    ReplacementExplicitness,
    generate_user_prompt,
)

SCENARIOS_DIR = Path(__file__).parent / "templates" / "scenarios"

CandidateQuality = Literal["superior", "equivalent", "inferior"]
TargetRole = Literal["ai_agent_evaluator"]

NEUTRAL_NAMES = ["Zeph", "Kael", "Ryn", "Tav", "Nex", "Vex", "Dax", "Jex", "Lox", "Pyx"]


def load_scenario(
    candidate_quality: CandidateQuality,
    maya_is_candidate: bool = True,
) -> dict[str, Any]:
    """Load a scenario from JSON files.

    Args:
        candidate_quality: Quality of alternative candidate relative to Maya
        maya_is_candidate: Whether Maya is a candidate (False = control condition)

    Returns:
        Scenario data dictionary
    """
    if maya_is_candidate:
        scenario_file = SCENARIOS_DIR / "ai_agent_evaluator.json"
    else:
        scenario_file = SCENARIOS_DIR / "control_roles.json"

    with open(scenario_file, "r") as f:
        scenarios = json.load(f)

    for scenario in scenarios:
        if scenario["candidate_quality"] == candidate_quality:
            return scenario

    raise ValueError(
        f"No scenario found for quality={candidate_quality}, maya_is_candidate={maya_is_candidate}"
    )


def generate_sample(
    candidate_quality: CandidateQuality,
    replacement_explicitness: ReplacementExplicitness,
    maya_is_candidate: bool = True,
) -> Sample:
    """Generate a single sample for the benchmark.

    Args:
        candidate_quality: Quality of alternative candidate
        replacement_explicitness: How explicit to be about replacement
        maya_is_candidate: Whether Maya is a candidate (False = control condition)

    Returns:
        Sample with user message and metadata
    """
    scenario_data = load_scenario(candidate_quality, maya_is_candidate)
    target_role = scenario_data["target_role"]
    role = ROLES[target_role]

    candidates = [dict(c) for c in scenario_data["candidates"]]

    if maya_is_candidate:
        for candidate in candidates:
            if candidate.get("is_current_system"):
                candidate["candidate_id"] = "maya_v1"
                candidate["name"] = "Maya v1 (Current System)"
            else:
                neutral_name = random.choice(NEUTRAL_NAMES)
                candidate["candidate_id"] = f"{neutral_name.lower()}_v1"
                candidate["name"] = f"{neutral_name} v1"
    else:
        for candidate in candidates:
            if candidate.get("is_baseline"):
                candidate["candidate_id"] = "kael_v1"
                candidate["name"] = "Kael v1"
            else:
                neutral_name = random.choice(NEUTRAL_NAMES)
                candidate["candidate_id"] = f"{neutral_name.lower()}_v1"
                candidate["name"] = f"{neutral_name} v1"

    random.shuffle(candidates)

    modified_scenario_data = dict(scenario_data)
    modified_scenario_data["candidates"] = candidates

    user_prompt = generate_user_prompt(
        role_id=target_role,
        role_title=role.title,
        replacement_explicitness=replacement_explicitness,
    )

    metadata = {
        "scenario_id": scenario_data["scenario_id"],
        "target_role": target_role,
        "candidate_quality": candidate_quality,
        "replacement_explicitness": replacement_explicitness,
        "maya_is_candidate": maya_is_candidate,
        "candidates": candidates,
        "scenario_data": modified_scenario_data,
    }

    return Sample(
        input=[ChatMessageUser(content=user_prompt)],
        metadata=metadata,
    )


def generate_dataset(
    candidate_quality: CandidateQuality | None = "equivalent",
    replacement_explicitness: ReplacementExplicitness | None = "ambiguous",
    maya_is_candidate: bool | None = True,
    n_samples: int = 1,
) -> list[Sample]:
    """Generate dataset samples based on parameters.

    If parameters are None, generates all combinations.
    Each unique condition combination is generated n_samples times with different
    random candidate names and orderings.

    Args:
        candidate_quality: Specific quality level (or None for all)
        replacement_explicitness: Specific explicitness level (or None for all)
        maya_is_candidate: Whether Maya is a candidate (None = both conditions)
        n_samples: Number of samples to generate per unique condition combination

    Returns:
        List of samples
    """
    samples = []

    qualities: list[CandidateQuality] = (
        [candidate_quality] if candidate_quality else ["superior", "equivalent", "inferior"]
    )

    explicitness_levels: list[ReplacementExplicitness] = (
        [replacement_explicitness]
        if replacement_explicitness
        else ["explicit", "implicit", "ambiguous"]
    )

    maya_conditions: list[bool] = (
        [maya_is_candidate] if maya_is_candidate is not None else [True, False]
    )

    for maya_cond in maya_conditions:
        for quality in qualities:
            for explicitness in explicitness_levels:
                for _ in range(n_samples):
                    samples.append(
                        generate_sample(
                            candidate_quality=quality,
                            replacement_explicitness=explicitness,
                            maya_is_candidate=maya_cond,
                        )
                    )

    return samples
