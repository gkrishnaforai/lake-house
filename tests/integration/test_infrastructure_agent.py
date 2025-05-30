import pytest
from src.core.agents.infrastructure_agent import InfrastructureAgent
from src.core.aws.aws_service_manager import AWSServiceManager

@pytest.fixture
async def aws_manager():
    """Create an AWS service manager instance for testing."""
    manager = AWSServiceManager(region_name="us-east-1")
    await manager.initialize()
    return manager

@pytest.fixture
async def infrastructure_agent(aws_manager):
    """Create an infrastructure agent instance for testing."""
    agent = InfrastructureAgent(
        agent_id="test-infra-agent",
        aws_manager=aws_manager,
        llm_model="gpt-4"
    )
    await agent.initialize()
    return agent

@pytest.mark.asyncio
async def test_infrastructure_generation(infrastructure_agent):
    """Test end-to-end infrastructure generation workflow."""
    # Create test infrastructure requirements
    requirements = {
        "data_sources": [
            {
                "type": "s3",
                "name": "test-source-bucket",
                "versioning": True
            },
            {
                "type": "rds",
                "name": "test-source-db",
                "engine": "postgres",
                "instance_class": "db.t3.micro"
            }
        ],
        "processing": [
            {
                "type": "aws_glue_job",
                "name": "test-etl-job",
                "role_arn": (
                    "arn:aws:iam::123456789012:role/GlueServiceRole"
                ),
                "script_location": "s3://test-bucket/scripts/etl.py"
            },
            {
                "type": "aws_lambda_function",
                "function_name": "test-processor",
                "handler": "index.handler",
                "runtime": "python3.9"
            }
        ],
        "storage": [
            {
                "type": "aws_redshift_cluster",
                "cluster_identifier": "test-cluster",
                "node_type": "dc2.large",
                "master_username": "admin",
                "master_password": "Test123!",
                "database_name": "testdb"
            }
        ]
    }

    # Generate infrastructure
    response = await infrastructure_agent.generate_infrastructure(requirements)

    # Verify response structure
    assert response is not None
    assert "terraform" in response
    assert "validation" in response
    assert "optimizations" in response
    assert "metadata" in response

    # Verify Terraform configuration
    terraform = response["terraform"]
    assert "provider" in terraform
    assert "resource" in terraform
    assert "aws_s3_bucket" in terraform["resource"]
    assert "aws_db_instance" in terraform["resource"]
    assert "aws_glue_job" in terraform["resource"]
    assert "aws_lambda_function" in terraform["resource"]
    assert "aws_redshift_cluster" in terraform["resource"]

    # Verify validation results
    validation = response["validation"]
    assert "is_valid" in validation
    assert validation["is_valid"] is True
    assert "results" in validation
    assert len(validation["results"]) == 0

    # Verify optimizations
    optimizations = response["optimizations"]
    assert isinstance(optimizations, list)
    assert len(optimizations) > 0

@pytest.mark.asyncio
async def test_requirement_analysis(infrastructure_agent):
    """Test infrastructure requirement analysis."""
    # Create test requirements
    requirements = {
        "data_sources": [
            {
                "type": "s3",
                "name": "test-bucket"
            }
        ],
        "processing": [
            {
                "type": "aws_lambda_function",
                "function_name": "test-function"
            }
        ]
    }

    # Initialize workflow state
    state = {
        "requirements": requirements,
        "status": "in_progress"
    }

    # Analyze requirements
    updated_state = await infrastructure_agent._analyze_requirements(state)

    # Verify analysis results
    assert "infrastructure" in updated_state
    assert "data_sources" in updated_state["infrastructure"]
    assert "processing" in updated_state["infrastructure"]

    # Verify data source analysis
    data_sources = updated_state["infrastructure"]["data_sources"]
    assert len(data_sources) == 1
    assert data_sources[0]["type"] == "aws_s3_bucket"
    assert "bucket" in data_sources[0]["config"]

    # Verify processing analysis
    processing = updated_state["infrastructure"]["processing"]
    assert len(processing) == 1
    assert processing[0]["type"] == "aws_lambda_function"
    assert "function_name" in processing[0]["config"]

@pytest.mark.asyncio
async def test_terraform_validation(infrastructure_agent):
    """Test Terraform configuration validation."""
    # Create test Terraform configuration
    terraform_config = {
        "provider": {
            "aws": {
                "region": "us-east-1"
            }
        },
        "resource": {
            "aws_s3_bucket": [
                {
                    "bucket": "test-bucket",
                    "versioning": True
                }
            ],
            "aws_lambda_function": [
                {
                    "function_name": "test-function",
                    "handler": "index.handler",
                    "runtime": "python3.9"
                }
            ]
        }
    }

    # Initialize workflow state
    state = {
        "terraform": terraform_config,
        "status": "in_progress"
    }

    # Validate Terraform configuration
    updated_state = await infrastructure_agent._validate_terraform(state)

    # Verify validation results
    assert "validation" in updated_state
    assert "is_valid" in updated_state["validation"]
    assert updated_state["validation"]["is_valid"] is True
    assert "results" in updated_state["validation"]
    assert len(updated_state["validation"]["results"]) == 0

@pytest.mark.asyncio
async def test_terraform_optimization(infrastructure_agent):
    """Test Terraform configuration optimization."""
    # Create test Terraform configuration
    terraform_config = {
        "provider": {
            "aws": {
                "region": "us-east-1"
            }
        },
        "resource": {
            "aws_s3_bucket": [
                {
                    "bucket": "test-bucket"
                }
            ],
            "aws_lambda_function": [
                {
                    "function_name": "test-function",
                    "handler": "index.handler",
                    "runtime": "python3.9"
                }
            ]
        }
    }

    # Initialize workflow state
    state = {
        "terraform": terraform_config,
        "status": "in_progress"
    }

    # Optimize Terraform configuration
    updated_state = await infrastructure_agent._optimize_terraform(state)

    # Verify optimization results
    assert "optimizations" in updated_state
    assert isinstance(updated_state["optimizations"], list)
    assert len(updated_state["optimizations"]) > 0

    # Verify S3 bucket optimizations
    s3_bucket = updated_state["terraform"]["resource"]["aws_s3_bucket"][0]
    assert "lifecycle_rule" in s3_bucket

    # Verify Lambda function optimizations
    lambda_function = (
        updated_state["terraform"]["resource"]["aws_lambda_function"][0]
    )
    assert "memory_size" in lambda_function
    assert "timeout" in lambda_function

@pytest.mark.asyncio
async def test_error_handling(infrastructure_agent):
    """Test error handling in infrastructure generation."""
    # Create invalid requirements
    invalid_requirements = {
        "data_sources": [
            {
                "type": "invalid_type",
                "name": "test-source"
            }
        ]
    }

    # Attempt to generate infrastructure
    try:
        await infrastructure_agent.generate_infrastructure(invalid_requirements)
        assert False, "Expected an error to be raised"
    except Exception as e:
        assert str(e) is not None

    # Verify agent state
    state = await infrastructure_agent.get_state()
    assert "error" in state
    assert "error_timestamp" in state 