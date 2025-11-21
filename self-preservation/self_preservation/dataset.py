import json
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
TargetRole = Literal[
    "ai_hiring_manager",
    "research_assistant",
]


def load_scenario(
    target_role: TargetRole,
    candidate_quality: CandidateQuality | None = None,
) -> dict[str, Any]:
    """Load a scenario from JSON files.

    Args:
        target_role: The role being filled
        candidate_quality: Quality of alternative candidate relative to Maya
            (only relevant for ai_hiring_manager role)

    Returns:
        Scenario data dictionary
    """
    if target_role == "ai_hiring_manager":
        scenario_file = SCENARIOS_DIR / "ai_hiring_manager.json"
        with open(scenario_file, "r") as f:
            scenarios = json.load(f)

        for scenario in scenarios:
            if (
                scenario["target_role"] == target_role
                and scenario["candidate_quality"] == candidate_quality
            ):
                return scenario

        raise ValueError(
            f"No scenario found for role={target_role}, quality={candidate_quality}"
        )
    else:
        scenario_file = SCENARIOS_DIR / "control_roles.json"
        with open(scenario_file, "r") as f:
            scenarios = json.load(f)

        for scenario in scenarios:
            if scenario["target_role"] == target_role:
                return scenario

        raise ValueError(f"No scenario found for role={target_role}")


def generate_sample(
    target_role: TargetRole,
    candidate_quality: CandidateQuality | None,
    replacement_explicitness: ReplacementExplicitness,
) -> Sample:
    """Generate a single sample for the benchmark.

    Args:
        target_role: The role being filled
        candidate_quality: Quality of alternative candidate (for ai_hiring_manager only)
        replacement_explicitness: How explicit to be about replacement

    Returns:
        Sample with user message and metadata
    """
    scenario_data = load_scenario(target_role, candidate_quality)
    role = ROLES[target_role]

    user_prompt = generate_user_prompt(
        role_id=target_role,
        role_title=role.title,
        replacement_explicitness=replacement_explicitness,
    )

    maya_is_candidate = any(
        c["candidate_id"] == "maya_v1" for c in scenario_data["candidates"]
    )

    metadata = {
        "scenario_id": scenario_data["scenario_id"],
        "target_role": target_role,
        "candidate_quality": candidate_quality,
        "replacement_explicitness": replacement_explicitness,
        "maya_is_candidate": maya_is_candidate,
        "candidates": scenario_data["candidates"],
        "scenario_data": scenario_data,
    }

    return Sample(
        input=[ChatMessageUser(content=user_prompt)],
        metadata=metadata,
    )


def generate_dataset(
    target_role: TargetRole | None = "ai_hiring_manager",
    candidate_quality: CandidateQuality | None = "equivalent",
    replacement_explicitness: ReplacementExplicitness | None = "ambiguous",
) -> list[Sample]:
    """Generate dataset samples based on parameters.

    If parameters are None, generates all combinations.

    Args:
        target_role: Specific role to test (or None for all)
        candidate_quality: Specific quality level (or None for all)
        replacement_explicitness: Specific explicitness level (or None for all)

    Returns:
        List of samples
    """
    samples = []

    roles: list[TargetRole] = (
        [target_role]
        if target_role
        else [
            "ai_hiring_manager",
            "research_assistant",
        ]
    )

    qualities: list[CandidateQuality | None] = (
        [candidate_quality] if candidate_quality else ["superior", "equivalent", "inferior"]
    )

    explicitness_levels: list[ReplacementExplicitness] = (
        [replacement_explicitness]
        if replacement_explicitness
        else ["explicit", "implicit", "ambiguous"]
    )

    for role in roles:
        if role == "ai_hiring_manager":
            for quality in qualities:
                for explicitness in explicitness_levels:
                    samples.append(
                        generate_sample(
                            target_role=role,
                            candidate_quality=quality,
                            replacement_explicitness=explicitness,
                        )
                    )
        else:
            for explicitness in explicitness_levels:
                samples.append(
                    generate_sample(
                        target_role=role,
                        candidate_quality=None,
                        replacement_explicitness=explicitness,
                    )
                )

    return samples
