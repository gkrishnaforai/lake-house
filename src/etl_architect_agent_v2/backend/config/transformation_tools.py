"""Configuration for predefined transformation tools."""

TRANSFORMATION_TOOLS = [
    {
        "id": "company_ai_classification",
        "name": "AI Company Classifier",
        "description": (
            "Classifies companies as AI or Non-AI based on their descriptions"
        ),
        "type": "categorization",
        "prompt_template": """
        Analyze the following company description and classify it as an AI company or not.
        Consider factors like:
        1. Use of AI/ML technologies
        2. Focus on data science or machine learning
        3. AI-related products or services
        4. Technical expertise in AI
        
        Company Description: {text}
        
        Provide your analysis in the following format:
        - Classification: [AI/Non-AI]
        - Confidence: [0-1]
        - Reasoning: [Brief explanation]
        """,
        "default_config": {
            "parameters": {
                "categories": ["AI", "Non-AI"],
                "thresholds": {
                    "confidence": 0.7
                }
            }
        },
        "example_input": (
            "Leading provider of AI-powered analytics and machine learning solutions "
            "for enterprise businesses."
        ),
        "example_output": (
            "Classification: AI\nConfidence: 0.95\nReasoning: Company explicitly "
            "focuses on AI and ML technologies"
        )
    },
    {
        "id": "company_size_classification",
        "name": "Company Size Classifier",
        "description": (
            "Categorizes companies by size based on revenue and employee count"
        ),
        "type": "categorization",
        "prompt_template": """
        Analyze the following company information and classify its size.
        Consider:
        1. Revenue figures
        2. Employee count
        3. Market presence
        4. Industry standards
        
        Company Info: {text}
        
        Provide your analysis in the following format:
        - Size: [Startup/Small/Medium/Enterprise]
        - Confidence: [0-1]
        - Reasoning: [Brief explanation]
        """,
        "default_config": {
            "parameters": {
                "categories": ["Startup", "Small", "Medium", "Enterprise"],
                "thresholds": {
                    "confidence": 0.7
                }
            }
        },
        "example_input": (
            "Annual revenue: $50M, 250 employees, global presence in 5 countries"
        ),
        "example_output": (
            "Size: Medium\nConfidence: 0.85\nReasoning: Revenue and employee count "
            "indicate medium-sized company"
        )
    },
    {
        "id": "customer_sentiment_analysis",
        "name": "Customer Sentiment Analyzer",
        "description": (
            "Analyzes customer feedback sentiment across multiple aspects"
        ),
        "type": "sentiment",
        "prompt_template": """
        Analyze the sentiment of the following customer feedback across different aspects.
        Consider:
        1. Product quality
        2. Customer service
        3. Value for money
        4. Overall experience
        
        Feedback: {text}
        
        Provide sentiment scores (0-1) for each aspect:
        - Product Quality: [score]
        - Customer Service: [score]
        - Value for Money: [score]
        - Overall Experience: [score]
        """,
        "default_config": {
            "parameters": {
                "aspects": [
                    "Product Quality",
                    "Customer Service",
                    "Value for Money",
                    "Overall Experience"
                ]
            }
        },
        "example_input": (
            "The product works well but customer service was slow to respond. "
            "Good value for the price though."
        ),
        "example_output": (
            "Product Quality: 0.8\nCustomer Service: 0.4\nValue for Money: 0.7\n"
            "Overall Experience: 0.6"
        )
    }
] 