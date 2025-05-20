"""Tests for the transformation agent."""

import pytest
from langchain_openai import ChatOpenAI
from core.llm.manager import LLMManager
from src.core.sql.sql_state import TransformationOutput
from etl_architect_agent_v2.agents.transformation.transformation_agent import (
    TransformationAgent,
    TransformationConfig,
    SentimentConfig,
    CategorizationConfig
)


@pytest.fixture
def llm():
    """Create LLM instance."""
    return ChatOpenAI(temperature=0)


@pytest.fixture
def llm_manager():
    """Create LLM manager instance."""
    return LLMManager()


@pytest.fixture
def transformation_agent(llm_manager):
    """Create a transformation agent instance."""
    return TransformationAgent(llm_manager)


@pytest.mark.asyncio
async def test_restaurant_review_sentiment_analysis(transformation_agent):
    """Test sentiment analysis on restaurant reviews with custom criteria."""
    # Sample restaurant review data
    data = [
        {
            "review_id": "R1",
            "restaurant": "Tasty Bites",
            "rating": 4,
            "comment": (
                "The food was amazing! The service was excellent and "
                "the atmosphere was perfect."
            ),
            "date": "2024-03-15"
        },
        {
            "review_id": "R2",
            "restaurant": "Tasty Bites",
            "rating": 2,
            "comment": (
                "Food was cold and service was slow. "
                "Not worth the price."
            ),
            "date": "2024-03-14"
        },
        {
            "review_id": "R3",
            "restaurant": "Tasty Bites",
            "rating": 3,
            "comment": (
                "Decent food but nothing special. "
                "Service was okay."
            ),
            "date": "2024-03-13"
        }
    ]

    # Custom sentiment analysis configuration
    config = TransformationConfig(
        sentiment=SentimentConfig(
            positive_threshold=0.7,  # More strict positive threshold
            negative_threshold=0.3,  # More strict negative threshold
            include_score=True
        )
    )

    # Custom metadata for restaurant-specific analysis
    metadata = {
        "analysis_type": "restaurant_review",
        "criteria": {
            "food_quality": True,
            "service": True,
            "value_for_money": True,
            "ambiance": True
        },
        "aspects": [
            "food",
            "service",
            "price",
            "ambiance"
        ]
    }

    # Apply transformation
    result = await transformation_agent.apply_transformation(
        data=data,
        transformation_type="sentiment",
        config=config,
        metadata=metadata
    )

    # Verify result
    assert isinstance(result, TransformationOutput)
    assert len(result.transformed_data) == 3
    
    # Verify first review (positive)
    first_review = result.transformed_data[0]
    assert first_review["original"]["review_id"] == "R1"
    assert first_review["transformed"]["overall_sentiment"] == "positive"
    assert first_review["transformed"]["sentiment_score"] > 0.7
    assert "aspect_sentiments" in first_review["transformed"]
    assert (
        first_review["transformed"]["aspect_sentiments"]["food"]
        ["sentiment"] == "positive"
    )
    
    # Verify second review (negative)
    second_review = result.transformed_data[1]
    assert second_review["original"]["review_id"] == "R2"
    assert second_review["transformed"]["overall_sentiment"] == "negative"
    assert second_review["transformed"]["sentiment_score"] < 0.3
    assert "aspect_sentiments" in second_review["transformed"]
    assert (
        second_review["transformed"]["aspect_sentiments"]["price"]
        ["sentiment"] == "negative"
    )
    
    # Verify third review (neutral)
    third_review = result.transformed_data[2]
    assert third_review["original"]["review_id"] == "R3"
    assert third_review["transformed"]["overall_sentiment"] == "neutral"
    assert 0.3 <= third_review["transformed"]["sentiment_score"] <= 0.7
    
    # Verify metadata
    assert result.metadata["transformation_type"] == "sentiment"
    assert result.metadata["total_records"] == 3
    assert result.metadata["successful_transformations"] == 3
    assert "analysis_summary" in result.metadata
    assert result.metadata["analysis_summary"]["positive_reviews"] == 1
    assert result.metadata["analysis_summary"]["negative_reviews"] == 1
    assert result.metadata["analysis_summary"]["neutral_reviews"] == 1


@pytest.mark.asyncio
async def test_company_categorization(transformation_agent):
    """Test company categorization with configurable LLM prompts."""
    # Sample company data
    data = [
        {
            "company_id": "C1",
            "name": "TechAI Solutions",
            "description": (
                "Leading provider of AI-powered analytics and machine "
                "learning solutions for enterprise businesses. We "
                "specialize in natural language processing and computer "
                "vision technologies."
            ),
            "revenue": "50M",
            "employees": 250,
            "city": "San Francisco",
            "state": "CA"
        },
        {
            "company_id": "C2",
            "name": "Green Energy Corp",
            "description": (
                "Renewable energy company focused on solar and wind "
                "power solutions. We provide sustainable energy "
                "alternatives for residential and commercial properties."
            ),
            "revenue": "30M",
            "employees": 150,
            "city": "Austin",
            "state": "TX"
        },
        {
            "company_id": "C3",
            "name": "DataFlow Systems",
            "description": (
                "Enterprise software company providing data integration "
                "and ETL solutions. Our platform helps businesses "
                "streamline their data workflows."
            ),
            "revenue": "25M",
            "employees": 120,
            "city": "Boston",
            "state": "MA"
        }
    ]

    # Custom categorization configuration
    config = TransformationConfig(
        categorization=CategorizationConfig(
            categories=["AI Company", "Non-AI Company"],
            allow_unknown=True,
            confidence_threshold=0.7
        )
    )

    # Custom metadata with LLM prompt template
    metadata = {
        "analysis_type": "company_categorization",
        "llm_prompt_template": """
        Analyze if the company is AI-focused based on these criteria:
        1. Primary focus on AI/ML technologies
        2. Mentions of specific AI technologies (NLP, CV, ML, etc.)
        3. AI-related products or services
        4. AI research or development focus
        
        Company Description: {description}
        
        Return categorization in this format:
        {
            "is_ai_company": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "explanation",
            "ai_technologies": ["list", "of", "technologies"],
            "ai_focus_areas": ["list", "of", "focus", "areas"]
        }
        """,
        "criteria": {
            "ai_technologies": True,
            "ai_products": True,
            "ai_research": True,
            "ai_services": True
        }
    }

    # Apply transformation
    result = await transformation_agent.apply_transformation(
        data=data,
        transformation_type="categorization",
        config=config,
        metadata=metadata
    )

    # Verify result
    assert isinstance(result, TransformationOutput)
    assert len(result.transformed_data) == 3
    
    # Verify AI company
    ai_company = result.transformed_data[0]
    assert ai_company["original"]["company_id"] == "C1"
    assert ai_company["transformed"]["category"] == "AI Company"
    assert ai_company["transformed"]["confidence"] > 0.7
    assert ai_company["transformed"]["details"]["is_ai_company"] is True
    assert len(ai_company["transformed"]["details"]["ai_technologies"]) > 0
    
    # Verify non-AI company
    non_ai_company = result.transformed_data[1]
    assert non_ai_company["original"]["company_id"] == "C2"
    assert non_ai_company["transformed"]["category"] == "Non-AI Company"
    assert non_ai_company["transformed"]["confidence"] > 0.7
    assert non_ai_company["transformed"]["details"]["is_ai_company"] is False
    assert (
        len(non_ai_company["transformed"]["details"]["ai_technologies"]) == 0
    )
    
    # Verify metadata
    assert result.metadata["transformation_type"] == "categorization"
    assert result.metadata["total_records"] == 3
    assert result.metadata["successful_transformations"] == 3
    assert "analysis_summary" in result.metadata
    assert result.metadata["analysis_summary"]["ai_companies"] == 1
    assert result.metadata["analysis_summary"]["non_ai_companies"] == 2
    assert (
        result.metadata["analysis_summary"]["average_confidence"] > 0.7
    )
    assert (
        len(result.metadata["analysis_summary"]["common_ai_technologies"]) > 0
    ) 