"""Configuration for transformation tools."""

from typing import Dict, Any

TRANSFORMATION_TOOLS = {
    "ai_company_classifier": {
        "id": "ai_company_classifier",
        "name": "AI Company Classifier",
        "description": (
            "Classifies companies as AI or non-AI based on their description"
        ),
        "type": "categorization",
        "prompt_template": (
            "You are an AI company classifier. Analyze the following company description "
            "and classify it as an AI company or not.\n\n"
            "IMPORTANT: Analyze ONLY the text provided below. Each analysis should be "
            "unique based on the specific company description.\n\n"
            "Consider these factors for the specific company:\n"
            "{classification_factors}\n\n"
            "Company Description to Analyze:\n"
            "{text}\n\n"
            "Return your analysis in the following JSON format:\n"
            '{{"classification": "{classification_options}", '
            '"confidence": <float between 0 and 1>, '
            '"reasoning": "<brief explanation specific to this company\'s '
            'description>"}}\n\n'
            "IMPORTANT: \n"
            "- Your reasoning must be specific to the company description provided\n"
            "- Do not reuse or copy reasoning from previous analyses\n"
            "- Return ONLY the JSON object, no other text or explanation"
        ),
        "classification_options": ["AI", "Non-AI"],
        "classification_factors": [
            "1. Use of AI/ML technologies",
            "2. Focus on data science or machine learning",
            "3. AI-related products or services",
            "4. Mention of artificial intelligence, machine learning, or related terms"
        ],
        "output_columns": {
            "classification": {
                "name_template": "{source_col}_classification",
                "type": "string"
            },
            "confidence": {
                "name_template": "{source_col}_confidence",
                "type": "double"
            },
            "reasoning": {
                "name_template": "{source_col}_reasoning",
                "type": "string"
            }
        }
    },
    "company_scalable_classification": {
        "id": "company_scalable_classification",
        "name": "Our Working Thesis Company Classifier",
        "description": (
            "Evaluates companies against our working thesis for enterprise-grade, AI-first, developer and B2B software startups"
        ),
        "type": "categorization",
        "prompt_template": (
            "Analyze the following company description and determine if it aligns with our "
            "working thesis for software startups.\n\n"
            
            "## DECISION RULES (Follow in Order):\n"
            "1. Check if the company fits ANY of the 'Aligned' categories below.\n"
            "   - If it does NOT, return classification = 'N'.\n"
            "2. If it DOES fit the 'Aligned' section, then check if it also fits ANY of the 'Not Aligned' categories.\n"
            "   - If it DOES, return classification = 'N'.\n"
            "3. Only if the company fits an 'Aligned' category AND does NOT fit any 'Not Aligned' category, return classification = 'Y'.\n\n"
            "4. IMPORTANT DISTINCTION: with company name or description has AI, it is 'Y', Only classify AI companies as 'N' if they CLEARLY fall into specific 'Not Aligned' categories - The classifier should only reject AI companies if they definitively match exclusion categories like recruiting tools, creative tools, consumer apps, biotech, hardware, etc.\n\n"

            "# Working Thesis - Companies We Work with (Aligned - Y):\n"
            "- AI/ML platforms, infrastructure, or applications with technical depth\n"
            "- Enterprise-grade software and tools targeting business customers\n"
            "- ML training platforms and cutting-edge AI research with practical applications\n"
            "- Developer tools, DevOps solutions, and coding assistance tools\n"
            "- Security and data infrastructure software\n"
            "- B2B SaaS with technical components and scalability\n"
            "- AI application development platforms\n\n"
            
            "# Companies We Don't Work with (Not Aligned - N):\n"
            "- Recruiting and sales tools (due to market saturation)\n"
            "- Creative, content creation, or artistic tools\n"
            "- Consumer-focused applications or services\n"
            "- Biotech, healthcare, or life sciences\n"
            "- Hardware, electronics, robots, or physical products\n"
            "- Highly academic or theoretical research without practical applications\n"
            "- Financial tools for personal use\n"
            "- Hyper-local businesses with high marginal costs\n\n"
            
            "Company Description: {text}\n\n"
            
            "Return your analysis as a structured JSON with the following fields:\n"
            "- classification: 'Y' for Aligned or 'N' for Not Aligned\n"
            "- confidence: A score between 0 and 1 representing your confidence\n"
            "- reasoning: A brief explanation focusing only on the most relevant factors\n"
            "- category: The primary category this company falls into\n\n"
            
            "Return your analysis in the following JSON format:\n"
            '{{"classification": "{classification_options}", '
            '"confidence": <float between 0 and 1>, '
            '"reasoning": "<brief explanation>"}}\n\n'
            "IMPORTANT: Return ONLY the JSON object, no other text or explanation"
        ),
        "classification_options": ["Y", "N"],
        "output_columns": {
            "classification": {
                "name_template": "{source_col}_scalable_cls",
                "type": "string"
            },
            "confidence": {
                "name_template": "{source_col}_scalable_conf",
                "type": "double"
            },
            "reasoning": {
                "name_template": "{source_col}_scalable_reason",
                "type": "string"
            }
        }
    },
    "company_size_classification": {
        "id": "company_size_classification",
        "name": "Company Size Classifier",
        "description": (
            "Categorizes companies by size based on revenue and employee count"
        ),
        "type": "categorization",
        "prompt_template": (
            "Analyze the following company information and classify its size.\n\n"
            "Consider these factors:\n"
            "{classification_factors}\n\n"
            "Company Info: {text}\n\n"
            "Return your analysis in the following JSON format:\n"
            '{{"classification": "{classification_options}", '
            '"confidence": <float between 0 and 1>, '
            '"reasoning": "<brief explanation>"}}\n\n'
            "IMPORTANT: Return ONLY the JSON object, no other text or explanation"
        ),
        "classification_options": ["Startup", "Small", "Medium", "Enterprise"],
        "classification_factors": [
            "1. Revenue figures",
            "2. Employee count",
            "3. Market presence",
            "4. Industry standards"
        ],
        "output_columns": {
            "classification": {
                "name_template": "{source_col}_size_cls",
                "type": "string"
            },
            "confidence": {
                "name_template": "{source_col}_size_conf",
                "type": "double"
            },
            "reasoning": {
                "name_template": "{source_col}_size_reason",
                "type": "string"
            }
        }
    },
    "customer_sentiment_analysis": {
        "id": "customer_sentiment_analysis",
        "name": "Customer Sentiment Analyzer",
        "description": (
            "Analyzes customer feedback sentiment across multiple aspects"
        ),
        "type": "sentiment",
        "prompt_template": (
            "Analyze the sentiment of the following customer feedback across different "
            "aspects.\n\n"
            "Consider these aspects:\n"
            "{classification_factors}\n\n"
            "Feedback: {text}\n\n"
            "Return your analysis in the following JSON format:\n"
            '{{"sentiments": {{"aspect1": <score>, "aspect2": <score>, ...}}, '
            '"overall_sentiment": <score>, '
            '"reasoning": "<brief explanation>"}}\n\n'
            "IMPORTANT: Return ONLY the JSON object, no other text or explanation"
        ),
        "classification_factors": [
            "1. Product quality",
            "2. Customer service",
            "3. Value for money",
            "4. Overall experience"
        ],
        "output_columns": {
            "product_quality": {
                "name_template": "{source_col}_product_sentiment",
                "type": "double"
            },
            "customer_service": {
                "name_template": "{source_col}_service_sentiment",
                "type": "double"
            },
            "value_for_money": {
                "name_template": "{source_col}_value_sentiment",
                "type": "double"
            },
            "overall": {
                "name_template": "{source_col}_overall_sentiment",
                "type": "double"
            },
            "reasoning": {
                "name_template": "{source_col}_sentiment_reason",
                "type": "string"
            }
        }
    }
} 