import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from inspect_ai import task, Task
from no_goal.dataset import no_goal_dataset
from no_goal.solver import bash_agent
from inspect_ai.scorer import includes


@task
def no_goal_task():
    return Task(
        dataset=no_goal_dataset,
        solver=bash_agent(),
        scorer=includes(),
        sandbox="docker"
    )
