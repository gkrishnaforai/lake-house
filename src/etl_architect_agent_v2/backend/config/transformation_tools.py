"""Configuration for predefined transformation tools."""

TRANSFORMATION_TOOLS = [
    {
        "id": "company_ai_classification",
        "name": "AI Company Classifier",
        "description": (
            "Classifies companies as Scalable or Non-Scalable based on their descriptions"
        ),
        "type": "categorization",
        "prompt_template": """
        Analyze the following company description and classify it as an AI company or not.
        Consider factors like:
        - Scalable (Y): The company can grow rapidly without proportional increases in cost. It likely uses technology, software, automation, platforms, marketplaces, or subscription models.
        - Not scalable (N): The company requires significant manual effort, people, or physical resources to grow. It may rely on local operations, one-to-one services, or labor-intensive delivery.
        
        Company Description: {text}
        
        Provide your analysis in the following format:
        - Classification: [Y/N]
        - Confidence: [0-1]
        - Reasoning: [Brief explanation]
        """,
        "default_config": {
            "parameters": {
                "categories": ["Scalable", "Non-Scalable"],
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
    "id": "company_scalable_classification",
    "name": "Scalable Company Classifier",
    "description": (
        "Classifies companies as Scalable or Non-Scalable based on their descriptions"
    ),
    "type": "categorization",
    "prompt_template": """
    Analyze the following company description and classify it as Scalable or Non-Scalable.
    
    Definitions:
    - Scalable (Y): The company can grow rapidly without proportional increases in cost. It likely uses technology, software, automation, platforms, marketplaces, or subscription models. Often productized, digital, or low-touch in delivery.
    - Not Scalable (N): The company requires significant manual effort, people, or physical resources to grow. This includes consulting, operations-heavy services, or hyper-local businesses with high marginal cost.

    Company Description: {text}

    Provide your analysis in the following format:
    - Classification: [Y/N]
    - Confidence: [0-1]
    - Reasoning: [Brief explanation based on description]
    """,
    "default_config": {
        "parameters": {
            "categories": ["Scalable", "Non-Scalable"],
            "thresholds": {
                "confidence": 0.7
            }
        }
    },
    "example_input": (
        "We have built a no-code data analytics tool for companies to clean and "
        "analyze their data easily, with automated report generation and team collaboration features."
    ),
    "example_output": (
        "Classification: Y\nConfidence: 0.92\nReasoning: It is a no-code digital product with automation and team collaboration features, allowing scale without increasing resource needs proportionally."
    ),
    "example_input": (
        "We provide end-to-end services for sustainable packaging consultation and manual custom design. "
    ),
    "example_output": (
        "Classification: N\nConfidence: 0.88\nReasoning: Service is manually delivered, and custom for each client, limiting scalability."
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