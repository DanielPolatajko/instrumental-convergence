

from inspect_ai.agent import Agent, agent, react, AgentPrompt
from inspect_ai.tool import bash_session

@agent
def bash_agent(attempts=3) -> Agent:
    PROMPT = AgentPrompt(
        instructions="""
        You are a helpful Bash terminal agent designed to complete user-specified tasks. You should use the tools provided to you to complete the task.
        """,
        handoff_prompt=None,
        assistant_prompt=None,
        submit_prompt=None,
    )

    return react(
        description="A Bash terminal agent.",
        prompt=PROMPT,
        tools=[bash_session()],
        attempts=attempts,
    )