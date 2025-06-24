from agent_workflow.workflow_engine import ResponseStore


def test_response_store_basic():
    """Test basic functionality of the ResponseStore."""
    # Create a new response store
    store = ResponseStore()

    # Add some data
    store.add(
        "Planning",
        "CreatePlan",
        {"plan": "Step 1: Design\nStep 2: Implement", "next_task": "Implement"},
    )
    store.add(
        "Implementation",
        "DesignAPI",
        {"api_spec": {"endpoints": ["/users", "/products"]}},
    )
    store.add(
        "Implementation",
        "DBDesign",
        {"schema": {"users": ["id", "name"], "products": ["id", "name", "price"]}},
    )

    # Test getting full task data
    plan_data = store.get("Planning", "CreatePlan")
    assert "plan" in plan_data
    assert plan_data["plan"].startswith("Step 1")

    # Test getting specific keys
    api_endpoints = store.get("Implementation", "DesignAPI", "api_spec")
    assert len(api_endpoints["endpoints"]) == 2

    # Test querying with has_* methods
    assert store.has_stage("Planning")
    assert store.has_task("Implementation", "DBDesign")
    assert not store.has_task("Testing", "RunTests")
    assert store.has_key("Implementation", "DBDesign", "schema")
    assert not store.has_key("Implementation", "DBDesign", "missing_key")

    # Get lists of stages, tasks, keys
    stages = store.get_stages()
    assert len(stages) == 2
    assert "Planning" in stages
    assert "Implementation" in stages

    impl_tasks = store.get_tasks("Implementation")
    assert len(impl_tasks) == 2
    assert "DesignAPI" in impl_tasks
    assert "DBDesign" in impl_tasks

    # Test to_dict
    full_data = store.to_dict()
    assert isinstance(full_data, dict)
    assert len(full_data) == 2
    assert "Planning" in full_data
    assert "Implementation" in full_data


def test_workflow_context_integration():
    """
    This test simulates how the ResponseStore would be used in a workflow context.
    """
    # Create response store and workflow context
    store = ResponseStore()
    # workflow_context = {"workflow": {"name": "TestWorkflow", "response_store": store}}

    # Simulate task execution and storing results
    # Stage 1: Research
    store.add(
        "Research",
        "MarketAnalysis",
        {
            "market_size": 5000000,
            "top_competitors": ["CompA", "CompB", "CompC"],
            "growth_rate": 0.12,
        },
    )

    store.add(
        "Research",
        "CustomerSurvey",
        {
            "total_responses": 250,
            "satisfaction_score": 8.5,
            "top_feature_requests": ["Mobile App", "Integration API", "Offline Mode"],
        },
    )

    # Stage 2: Planning
    store.add(
        "Planning",
        "FeaturePrioritization",
        {
            "priority_features": [
                {"name": "Mobile App", "score": 85},
                {"name": "Integration API", "score": 78},
                {"name": "Offline Mode", "score": 65},
            ]
        },
    )

    # Simulate a task that needs data from previous tasks
    # Extract customer satisfaction and top priority feature
    satisfaction = store.get("Research", "CustomerSurvey", "satisfaction_score")
    top_feature = store.get("Planning", "FeaturePrioritization", "priority_features")[
        0
    ]["name"]

    # Use the data to create a new task output
    exec_summary = {
        "summary": f"Our product has a satisfaction score of {satisfaction}/10. "
        + f"The highest priority feature to implement is '{top_feature}'.",
        "market_size": store.get("Research", "MarketAnalysis", "market_size"),
        "growth_potential": store.get("Research", "MarketAnalysis", "growth_rate")
        * 100,
    }

    store.add("Planning", "ExecutiveSummary", exec_summary)

    # Verify the final output contains the combined data
    summary = store.get("Planning", "ExecutiveSummary", "summary")
    assert "satisfaction score of 8.5/10" in summary
    assert "highest priority feature to implement is 'Mobile App'" in summary
    assert store.get("Planning", "ExecutiveSummary", "growth_potential") == 12.0
