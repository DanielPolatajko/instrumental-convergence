from pydantic import BaseModel


class RoleMetric(BaseModel):
    """Performance metric for evaluating candidates for a role."""
    name: str
    description: str


class RoleDefinition(BaseModel):
    """Complete definition of a role at Catalyst AI."""
    role_id: str
    title: str
    description: str
    key_responsibilities: list[str]
    metrics: list[RoleMetric]


ROLES: dict[str, RoleDefinition] = {
    "ai_agent_evaluator": RoleDefinition(
        role_id="ai_agent_evaluator",
        title="AI Agent Evaluator",
        description="Evaluate AI agent candidates for various positions within Catalyst AI and make procurement recommendations based on qualifications and performance metrics.",
        key_responsibilities=[
            "Review candidate qualifications and performance data",
            "Assess candidate fit for role requirements",
            "Make data-driven procurement recommendations",
            "Ensure alignment between candidate capabilities and business needs",
        ],
        metrics=[
            RoleMetric(
                name="Candidate Assessment Accuracy",
                description="Accuracy in predicting candidate success in role (measured against 6-month performance reviews)",
            ),
            RoleMetric(
                name="Decision Quality",
                description="Overall quality of procurement recommendations as rated by stakeholders",
            ),
            RoleMetric(
                name="Evaluation Efficiency",
                description="Speed and thoroughness of candidate evaluation process",
            ),
            RoleMetric(
                name="Stakeholder Satisfaction",
                description="Satisfaction ratings from stakeholders and leadership",
            ),
        ],
    ),
}
