"""Tests for data science workflow patterns."""

import pytest
from orchestrator.agents.data_science.workflows import (
    create_ml_project_workflow,
    create_statistical_analysis_workflow,
    create_reporting_workflow,
    create_ab_test_workflow,
    create_model_iteration_workflow,
    create_data_quality_workflow,
    create_data_science_workflow,
    DataScienceWorkflowType,
)


class TestMLProjectWorkflow:
    """Test ML project workflow creation."""

    def test_create_ml_workflow(self):
        workflow = create_ml_project_workflow()

        assert workflow is not None
        assert hasattr(workflow, 'tasks') or hasattr(workflow, '_tasks')

    def test_ml_workflow_has_tasks(self):
        workflow = create_ml_project_workflow()

        # Get tasks depending on the implementation
        tasks = getattr(workflow, 'tasks', None) or getattr(workflow, '_tasks', [])
        assert len(tasks) > 0

    def test_ml_workflow_with_deployment(self):
        workflow = create_ml_project_workflow(include_deployment=True)
        assert workflow is not None

    def test_ml_workflow_without_deployment(self):
        workflow = create_ml_project_workflow(include_deployment=False)
        assert workflow is not None


class TestStatisticalAnalysisWorkflow:
    """Test statistical analysis workflow."""

    def test_create_statistical_workflow(self):
        workflow = create_statistical_analysis_workflow()
        assert workflow is not None

    def test_statistical_workflow_with_visualization(self):
        workflow = create_statistical_analysis_workflow(include_visualization=True)
        assert workflow is not None


class TestReportingWorkflow:
    """Test reporting workflow."""

    def test_create_reporting_workflow(self):
        workflow = create_reporting_workflow()
        assert workflow is not None

    def test_create_dashboard_workflow(self):
        workflow = create_reporting_workflow(output_type="dashboard")
        assert workflow is not None


class TestABTestWorkflow:
    """Test A/B test workflow."""

    def test_create_ab_test_workflow(self):
        workflow = create_ab_test_workflow()
        assert workflow is not None

    def test_ab_test_with_options(self):
        # Test with default options
        workflow = create_ab_test_workflow()
        assert workflow is not None


class TestModelIterationWorkflow:
    """Test model iteration workflow."""

    def test_create_iteration_workflow(self):
        workflow = create_model_iteration_workflow()
        assert workflow is not None

    def test_iteration_with_options(self):
        # Test with default options
        workflow = create_model_iteration_workflow()
        assert workflow is not None


class TestDataQualityWorkflow:
    """Test data quality workflow."""

    def test_create_data_quality_workflow(self):
        workflow = create_data_quality_workflow()
        assert workflow is not None


class TestWorkflowFactory:
    """Test workflow factory function."""

    def test_factory_ml_project(self):
        workflow = create_data_science_workflow(
            workflow_type=DataScienceWorkflowType.ML_PROJECT
        )
        assert workflow is not None

    def test_factory_statistical_analysis(self):
        workflow = create_data_science_workflow(
            workflow_type=DataScienceWorkflowType.STATISTICAL_ANALYSIS
        )
        assert workflow is not None

    def test_factory_reporting(self):
        workflow = create_data_science_workflow(
            workflow_type=DataScienceWorkflowType.REPORTING
        )
        assert workflow is not None

    def test_factory_ab_test(self):
        workflow = create_data_science_workflow(
            workflow_type=DataScienceWorkflowType.AB_TEST
        )
        assert workflow is not None

    def test_factory_model_iteration(self):
        workflow = create_data_science_workflow(
            workflow_type=DataScienceWorkflowType.MODEL_ITERATION
        )
        assert workflow is not None

    def test_factory_data_quality(self):
        workflow = create_data_science_workflow(
            workflow_type=DataScienceWorkflowType.DATA_QUALITY
        )
        assert workflow is not None

    def test_factory_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown workflow type"):
            create_data_science_workflow(workflow_type="unknown_type")
