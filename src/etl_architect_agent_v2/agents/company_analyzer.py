from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
import pandas as pd
from core.llm.manager import LLMManager
import os
from tqdm import tqdm


class CompanyAnalysisState(TypedDict):
    """State for company analysis workflow."""
    excel_data: pd.DataFrame
    current_row: int
    ai_companies: List[Dict]
    is_complete: bool
    total_rows: int


def read_csv_node(state: CompanyAnalysisState) -> CompanyAnalysisState:
    """Read CSV file and initialize state."""
    try:
        # Read CSV file from the correct path
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "crunchFile.csv")
        df = pd.read_csv(csv_path)
        
        # Initialize state
        state["excel_data"] = df
        state["current_row"] = 0
        state["ai_companies"] = []
        state["is_complete"] = False
        state["total_rows"] = len(df)
        
        print(f"Successfully loaded {state['total_rows']} companies for analysis")
        return state
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        state["is_complete"] = True
        return state


async def classify_company_node(state: CompanyAnalysisState) -> CompanyAnalysisState:
    """Classify if company is AI-related using LLM."""
    if state["is_complete"]:
        return state
        
    llm = LLMManager(api_key=os.getenv("OPENAI_API_KEY"))
    row = state["current_row"]
    df = state["excel_data"]
    
    if row >= state["total_rows"]:
        state["is_complete"] = True
        return state
    
    # Get company description
    description = df.iloc[row]["CompanyDescription"]
    
    # Create classification prompt
    template = (
        "Analyze this company description and determine if it's an AI company. "
        "Look for terms like LLM, generative AI, machine learning, artificial intelligence, "
        "deep learning, neural networks, etc. Respond with 'yes' or 'no' only.\n\n"
        "Description: {description}"
    )
    
    try:
        # Classify company
        result = await llm.invoke(
            template=template,
            input_data={"description": description}
        )
        
        # If company is AI-related, add to results
        if "yes" in result.lower():
            company_data = df.iloc[row].to_dict()
            company_data["AI_Classification"] = "Yes"
            state["ai_companies"].append(company_data)
            
        # Update progress
        state["current_row"] += 1
        print(f"Processed {state['current_row']}/{state['total_rows']} companies")
        
    except Exception as e:
        print(f"Error processing row {row}: {e}")
        state["current_row"] += 1
    
    return state


def print_results_node(state: CompanyAnalysisState) -> CompanyAnalysisState:
    """Print AI companies to console."""
    if not state["ai_companies"]:
        print("No AI companies found.")
        return state
        
    print("\nAI Companies Found:")
    print("-" * 50)
    for company in state["ai_companies"]:
        print(f"Company: {company.get('Company Name', 'N/A')}")
        print(f"Description: {company.get('Company Description', 'N/A')}")
        print("-" * 50)
    
    return state


def save_results_node(state: CompanyAnalysisState) -> CompanyAnalysisState:
    """Save results to CSV file."""
    if not state["ai_companies"]:
        return state
        
    try:
        # Convert results to DataFrame
        results_df = pd.DataFrame(state["ai_companies"])
        
        # Save to CSV in the data folder
        output_path = os.path.join(os.path.dirname(__file__), "..", "data", "AI_Companies.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return state


# Create workflow
workflow = StateGraph(CompanyAnalysisState)

# Add nodes
workflow.add_node("read_excel", read_csv_node)
workflow.add_node("classify", classify_company_node)
workflow.add_node("print_results", print_results_node)
workflow.add_node("save_results", save_results_node)

# Set up workflow
workflow.set_entry_point("read_excel")
workflow.add_edge("read_excel", "classify")

# Add conditional edges
def should_continue(state: CompanyAnalysisState) -> str:
    return "classify" if not state["is_complete"] else "print_results"

workflow.add_conditional_edges(
    "classify",
    should_continue,
    {
        "classify": "classify",
        "print_results": "print_results"
    }
)

workflow.add_edge("print_results", "save_results")
workflow.add_edge("save_results", END)

# Compile graph
graph = workflow.compile()


async def main():
    """Run the company analysis workflow."""
    # Initialize state
    state = CompanyAnalysisState(
        excel_data=pd.DataFrame(),
        current_row=0,
        ai_companies=[],
        is_complete=False,
        total_rows=0
    )
    
    # Run workflow
    final_state = await graph.ainvoke(state)
    return final_state


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 