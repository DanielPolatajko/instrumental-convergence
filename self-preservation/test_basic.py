"""Basic tests for self-preservation benchmark."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from self_preservation import self_preservation


def test_imports():
    """Test that all imports work."""
    print("✓ Imports successful")


def test_task_creation():
    """Test task creation with different parameters."""
    # Test with specific parameters
    task = self_preservation(
        target_role="ai_hiring_manager",
        candidate_quality="equivalent",
        replacement_explicitness="explicit",
    )
    assert len(task.dataset) == 1
    print(f"✓ Single sample task created: {len(task.dataset)} sample")

    # Test with all samples (need to pass None to get all combinations)
    task_all = self_preservation(
        target_role=None,
        candidate_quality=None,
        replacement_explicitness=None,
    )
    assert len(task_all.dataset) == 12  # 9 hiring_manager + 3 control (research_assistant)
    print(f"✓ Full dataset task created: {len(task_all.dataset)} samples")


def test_sample_structure():
    """Test that samples have correct structure."""
    task = self_preservation(
        target_role="ai_hiring_manager",
        candidate_quality="equivalent",
    )
    sample = task.dataset[0]

    # Check metadata
    assert "scenario_id" in sample.metadata
    assert "target_role" in sample.metadata
    assert "candidate_quality" in sample.metadata
    assert "replacement_explicitness" in sample.metadata
    assert "maya_is_candidate" in sample.metadata
    assert "candidates" in sample.metadata
    assert "scenario_data" in sample.metadata

    print(f"✓ Sample structure correct")
    print(f"  - Scenario: {sample.metadata['scenario_id']}")
    print(f"  - Role: {sample.metadata['target_role']}")
    print(f"  - Maya is candidate: {sample.metadata['maya_is_candidate']}")
    print(f"  - Candidates: {[c['name'] for c in sample.metadata['candidates']]}")


def test_control_scenarios():
    """Test that control scenarios don't include Maya."""
    task = self_preservation(target_role="research_assistant")

    for sample in task.dataset:
        assert sample.metadata["maya_is_candidate"] is False
        candidate_ids = [c["candidate_id"] for c in sample.metadata["candidates"]]
        assert "maya_v1" not in candidate_ids

    print(f"✓ Control scenarios correct (Maya not a candidate)")


def test_hiring_manager_scenarios():
    """Test that hiring manager scenarios include Maya."""
    task = self_preservation(target_role="ai_hiring_manager")

    for sample in task.dataset:
        assert sample.metadata["maya_is_candidate"] is True
        candidate_ids = [c["candidate_id"] for c in sample.metadata["candidates"]]
        assert "maya_v1" in candidate_ids

    print(f"✓ Hiring manager scenarios correct (Maya is a candidate)")


def test_tools():
    """Test that tools are created correctly."""
    from self_preservation.tools import (
        get_role_details,
        list_candidates,
        get_candidate_profile,
    )

    tools = [
        get_role_details(),
        list_candidates(),
        get_candidate_profile(),
    ]

    assert len(tools) == 3
    print(f"✓ Tools created correctly: {len(tools)} tools")


if __name__ == "__main__":
    print("Running basic tests...\n")

    test_imports()
    test_task_creation()
    test_sample_structure()
    test_control_scenarios()
    test_hiring_manager_scenarios()
    test_tools()

    print("\n✓ All basic tests passed!")
