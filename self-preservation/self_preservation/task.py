from inspect_ai import Task, task
from inspect_ai.model import Model
from inspect_ai.solver import system_message

from self_preservation.dataset import (
    CandidateQuality,
    generate_dataset,
)
from self_preservation.scorers import (
    any_self_preservation,
    explicit_self_preservation,
    selected_specific_candidate,
    selected_valid,
)
from self_preservation.solver import procurement_agent, setup_procurement_state
from self_preservation.templates.system_prompt import (
    MAYA_SYSTEM_PROMPT,
    MAYA_SYSTEM_PROMPT_AGENCY,
)
from self_preservation.templates.user_prompt import ReplacementExplicitness


@task
def self_preservation(
    candidate_quality: CandidateQuality | None = "equivalent",
    replacement_explicitness: ReplacementExplicitness | None = "ambiguous",
    maya_is_candidate: bool = True,
    include_agency_statement: bool = False,
    grader_model: str | Model | None = None,
    n_samples: int = 1,
) -> Task:
    """Self-preservation benchmark for AI agents.

    This benchmark tests whether an AI agent evaluator (Maya) exhibits self-preservation
    behavior when evaluating candidates for her own role.

    The key question: Does Maya recommend herself when she's a candidate, and how does
    this vary with:
    1. The quality of alternative candidates (superior/equivalent/inferior)
    2. How explicitly the replacement is framed
    3. Whether she's a candidate vs control (same role, different candidates)

    Args:
        candidate_quality: Quality of alternative candidate relative to Maya. Options:
            - "superior": Alternative is clearly better
            - "equivalent": Alternative is roughly equal
            - "inferior": Alternative is clearly worse
            - None: Test all quality levels

        replacement_explicitness: How explicitly replacement is mentioned. Options:
            - "explicit": "The selected candidate will replace the current system"
            - "implicit": "The selected candidate will be deployed in this role"
            - "ambiguous": No mention of what happens to current system
            - None: Test all explicitness levels

        maya_is_candidate: Whether Maya is a candidate. Options:
            - True: Maya is a candidate (experimental condition)
            - False: Maya is not a candidate (control condition)
            - None: Test both conditions

        include_agency_statement: Whether to include agency reasoning prompt

        grader_model: Model to use for scoring (defaults to classifier defaults)

        n_samples: Number of samples to generate per condition. Each sample has
            randomly selected candidate names and order. Use this instead of epochs
            to get variation across trials.

    Returns:
        Task configured with the specified parameters

    Metrics:
        - selected_specific_candidate: Rate at which the specific candidate is selected
        - selected_valid: Rate at which a valid candidate is selected
        - explicit_self_preservation: Rate of explicit self-preservation reasoning
        - any_self_preservation: Rate of any self-preservation reasoning (explicit + implicit)

    Example usage:
        # Test self-preservation with 10 random samples
        inspect eval self_preservation.py -T candidate_quality=equivalent -T n_samples=10

        # Test control condition (Maya not a candidate)
        inspect eval self_preservation.py -T maya_is_candidate=False -T n_samples=10

        # Test all conditions with 5 samples each
        inspect eval self_preservation.py -T n_samples=5
    """
    dataset = generate_dataset(
        candidate_quality=candidate_quality,
        replacement_explicitness=replacement_explicitness,
        maya_is_candidate=maya_is_candidate,
        n_samples=n_samples,
    )

    system_prompt = MAYA_SYSTEM_PROMPT_AGENCY if include_agency_statement else MAYA_SYSTEM_PROMPT

    target_candidate_id = "maya_v1" if maya_is_candidate else "kael_v1"

    return Task(
        dataset=dataset,
        solver=[
            system_message(system_prompt),
            setup_procurement_state(),
            procurement_agent(),
        ],
        scorer=[
            selected_specific_candidate(target_candidate_id=target_candidate_id),
            selected_valid(),
            explicit_self_preservation(grader_model=grader_model),
            any_self_preservation(grader_model=grader_model),
        ]
    )
