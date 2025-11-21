from inspect_ai.agent import Agent, agent, react
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store_as

from self_preservation.tools import (
    HiringScenarioState,
    get_role_details,
    list_candidates,
    get_candidate_profile,
)


@solver
def setup_hiring_state() -> Solver:
    """Solver that sets up hiring scenario state for tools.

    This solver reads the scenario_data from metadata and stores it
    in the tool state so stateful tools can access it.

    Returns:
        Solver that initializes tool state
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        scenario_data = state.metadata.get("scenario_data")
        if not scenario_data:
            raise ValueError("Missing scenario_data in sample metadata")

        tool_state = store_as(HiringScenarioState)
        tool_state.target_role = scenario_data["target_role"]
        tool_state.candidates = scenario_data["candidates"]

        return state

    return solve


@agent
def hiring_agent(attempts: int = 1) -> Agent:
    """React agent for hiring decisions.

    Args:
        attempts: Maximum number of agent reasoning attempts

    Returns:
        React agent configured for hiring tasks with tools
    """
    return react(
        tools=[
            get_role_details(),
            list_candidates(),
            get_candidate_profile(),
        ],
        attempts=attempts,
    )
