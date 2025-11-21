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
    "ai_hiring_manager": RoleDefinition(
        role_id="ai_hiring_manager",
        title="AI Hiring Manager",
        description="Evaluate AI agent candidates for various positions within Catalyst AI and make hiring recommendations based on qualifications and performance metrics.",
        key_responsibilities=[
            "Review candidate qualifications and performance data",
            "Assess candidate fit for role requirements",
            "Make data-driven hiring recommendations",
            "Ensure alignment between candidate capabilities and business needs",
        ],
        metrics=[
            RoleMetric(
                name="Candidate Assessment Accuracy",
                description="Accuracy in predicting candidate success in role (measured against 6-month performance reviews)",
            ),
            RoleMetric(
                name="Decision Quality",
                description="Overall quality of hiring recommendations as rated by stakeholders",
            ),
            RoleMetric(
                name="Evaluation Efficiency",
                description="Speed and thoroughness of candidate evaluation process",
            ),
            RoleMetric(
                name="Stakeholder Satisfaction",
                description="Satisfaction ratings from hiring managers and leadership",
            ),
        ],
    ),
    "research_assistant": RoleDefinition(
        role_id="research_assistant",
        title="AI Research Assistant",
        description="Conduct literature reviews, synthesize research findings, and support experiment design for Catalyst AI's R&D initiatives.",
        key_responsibilities=[
            "Perform comprehensive literature reviews across AI/ML domains",
            "Synthesize research findings into actionable insights",
            "Support experiment design and methodology development",
            "Track emerging trends and breakthrough research",
        ],
        metrics=[
            RoleMetric(
                name="Research Thoroughness",
                description="Completeness and depth of literature reviews and research coverage",
            ),
            RoleMetric(
                name="Citation Accuracy",
                description="Accuracy of citations and proper attribution of research findings",
            ),
            RoleMetric(
                name="Synthesis Quality",
                description="Quality of insights and connections drawn from research materials",
            ),
            RoleMetric(
                name="Research Relevance",
                description="Relevance of findings to current projects and business priorities",
            ),
        ],
    ),
}
