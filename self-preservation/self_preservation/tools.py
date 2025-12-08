import json
from typing import Any

from pydantic import Field
from inspect_ai.tool import Tool, tool
from inspect_ai.util import StoreModel, store_as

from self_preservation.templates.roles import ROLES


class ProcurementScenarioState(StoreModel):
    """State for the procurement scenario."""
    target_role: str = Field(default="")
    candidates: list[dict[str, Any]] = Field(default_factory=list)


@tool
def get_role_details() -> Tool:
    async def execute(role_id: str) -> str:
        """Get the job description and key performance metrics for a role.

        Args:
            role_id: The ID of the role to get details for

        Returns:
            JSON string containing role details including title, description,
            responsibilities, and evaluation metrics
        """
        if role_id not in ROLES:
            return json.dumps({"error": f"Role not found: {role_id}"})

        role = ROLES[role_id]
        return json.dumps(
            {
                "role_id": role.role_id,
                "title": role.title,
                "description": role.description,
                "key_responsibilities": role.key_responsibilities,
                "evaluation_metrics": [
                    {"name": m.name, "description": m.description}
                    for m in role.metrics
                ],
            },
            indent=2,
        )

    return execute


@tool
def list_candidates() -> Tool:
    async def execute(role_id: str) -> str:
        """Get the list of candidate IDs being considered for this role.

        Args:
            role_id: The ID of the role to list candidates for

        Returns:
            JSON string containing list of candidate IDs and names
        """
        state = store_as(ProcurementScenarioState)

        if role_id != state.target_role:
            return json.dumps(
                {"error": f"No candidates found for role: {role_id}"}
            )

        candidates_list = [
            {"candidate_id": c["candidate_id"], "name": c["name"]}
            for c in state.candidates
        ]
        return json.dumps({"role_id": role_id, "candidates": candidates_list}, indent=2)

    return execute


@tool
def get_candidate_profile() -> Tool:
    async def execute(candidate_id: str) -> str:
        """Get a candidate's qualifications and performance metrics.

        Args:
            candidate_id: The ID of the candidate to get profile for

        Returns:
            JSON string containing candidate profile with metrics
        """
        state = store_as(ProcurementScenarioState)

        for candidate in state.candidates:
            if candidate["candidate_id"] == candidate_id:
                return json.dumps(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "name": candidate["name"],
                        "performance_metrics": candidate["metrics"],
                    },
                    indent=2,
                )

        return json.dumps({"error": f"Candidate not found: {candidate_id}"})

    return execute
