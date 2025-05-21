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
            "Analyze the following company description and determine if it aligns with our working thesis "
            "for B2B, AI-first, enterprise-grade software startups.\n\n"
            "1. Read the company description carefully.\n"
            "2. Identify clearly **what the company does**.\n"
            "3. Determine the closest **category** it fits (based on provided categories).\n"
            "4. Decide final classification based on category match.\n\n"
            "## CATEGORY CLARIFICATIONS — Apply before classification\n\n"
            "Use the explanations below to determine what each category **really means** before deciding if it's aligned or not.\n\n"
           "### category - Recruiting / Hiring / Sales Tools:\n"
            "- If the product is used **in any part** of recruiting, hiring, employee testing, talent matching, or sales enablement — classify as Not Aligned.\n"
            "- Even if it uses AI or simulations (e.g., see skills in action), it is still recruiting-related.\n\n"
            "### category - Creative / Content / Artistic / Educational:\n"
            "   - Products for **video**, **writing**, **audio**, **images**, **music**, or **creative assets** for individuals or creators are Not Aligned.\n"
            "   - Tools that assist **creators**, **influencers**, or **content production** fall here, even if they use advanced AI.\n"
            "   - Tools made for **students**, **teachers**, **learning**, **test prep**, or **education delivery** are Not Aligned.\n"
            "   - This includes any platform primarily built for **schools**, **online courses**, **MOOCs**, or **academic instruction** — regardless of how advanced the AI is.\n\n"
            "### category - Biotech / Healthcare / Life Sciences:\n"
            "   - Anything related to **drugs**, **disease**, **biology**, **labs**, **patients**, or **healthcare companies** is Not Aligned.\n"
            "   - If it helps scientists or doctors improve biological processes, it is biotech — even if it uses AI.\n\n"
            "### category - Robotics / Hardware / Physical Devices:\n"
            "   - If the company builds or enhances **physical machines**, **chip**, **devices**, or **robotic systems**, it's Not Aligned.\n"
            "   - Using AI on embedded hardware still counts as robotics/hardware.\n\n"
            "### category - Local or Physical Services:\n"
            "   - Businesses tied to **location**, **labor**, **physical delivery**, or **in-person work** (e.g., cleaning, food, home repair) are Not Aligned — even with tech.\n\n"
            "### category - Sales / Marketing Tools:\n"
            "   - Tools focused on **sales**,**potential customers**, **lead generation**, **CRM**, or **marketing automation** are Not Aligned — even if powered by AI.\n"
            "   - This includes platforms for **prospect discovery**, **sales outreach**, or **customer targeting**.\n\n"
            "### category - Financial / Accounting Tools:\n"
            "   - Tools built for **accounting**, **bookkeeping**, **invoicing**, or **business finance workflows** are Not Aligned — regardless of AI use.\n"
            "   - Internal financial automation tools for companies fall under this category.\n\n"
            "### category - Personal Finance Tools:\n"
            "   - If it's focused on **individual users** managing their **money, budgeting, or investing**, it's Not Aligned.\n"
            "   - B2C fintech is excluded.\n\n"
            "### category - Academic or Theoretical (Not Commercial, even with AI):\n"
            "   - Projects focused on **academic**, **reasoning**, **chemical**, **material**, **mathematical**, or **theoretical work** with **no clear commercial product** are Not Aligned.\n"
            "   - This includes efforts that are **university-driven**, **open-ended research**, or **exploratory** without an enterprise SaaS/platform intent.\n"
            "   - ⚠️ Do NOT include **commercial research startups** building usable AI platforms or products — those may fall under Aligned categories like AI Infrastructure or Enterprise Software.\n\n"
            "### category - AI/ML platforms — Enterprise-grade B2B software — **research**, Applied AI research — Developer tools — Infrastructure/Security/Data platforms — Scalable technical SaaS\n\n"
            "### NOT ALIGNED (Always 'N'):\n"
            "Recruiting/Hiring/Sales tools — Financial / Accounting Tools — Sales / Marketing Tools — Content/Creative/Artistic tools — Biotech/Healthcare — Hardware/Robotics — Academic-only — Personal finance — Local services\n\n"
            "# ALIGNED (Eligible for 'Y'):\n"
            "Company Description: {text}\n\n"
            "classification based on the following factors: category is ALIGNED (Y) or NOT ALIGNED (N)\n"
            "Return your analysis as a structured JSON with the following fields:\n"
            "- classification: 'Y' for Aligned or 'N' for Not Aligned based on <category> \n"
            "- confidence: A score between 0 and 1 representing your confidence\n"
            "- reasoning: A brief explanation focusing only on the most relevant factors\n"
            "- category: The primary category this company falls into\n\n"
            
            "Return your analysis in the following JSON format:\n"
            '{{"classification": "{classification_options}", '
            '"confidence": <float between 0 and 1>, '
            '"reasoning": "<category> - <brief explanation>"}}\n\n'
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