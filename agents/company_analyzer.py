import pandas as pd
from etl_workflow_state import ETLWorkflowState
from langgraph.graph import StateGraph, END
import asyncio
from core.llm.manager import LLMManager
import os


async def read_csv_file(state: ETLWorkflowState) -> ETLWorkflowState:
    """Read the CSV file and store it in the state."""
    try:
        # Read the CSV file
        df = pd.read_csv('data/crunchFile.csv', encoding='utf-8')
        
        # Clean column names by stripping quotes, whitespace and newlines
        df.columns = [
            col.strip().replace('\n', '').strip("'") 
            for col in df.columns
        ]
        
        # Print original columns for debugging
        print("Original columns:", df.columns.tolist())
        print("\nFirst row:", df.iloc[0].to_dict())
        
        # Store the DataFrame in the state
        state.data['companies_df'] = df
        state.data['current_index'] = 0
        print(f"Loaded {len(df)} companies for analysis")
        
        return state
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        raise


async def classify_company_node(state: ETLWorkflowState) -> ETLWorkflowState:
    """Classify a company based on its description."""
    try:
        df = state.data["companies_df"]
        current_index = state.data["current_index"]
        
        if current_index >= len(df):
            return {"type": "end"}
            
        # Get the current row
        current_row = df.iloc[current_index]
        
        # Extract company information
        company_name = current_row['Company Name']
        company_description = current_row['Company Description']
        sector = current_row['Sector']
        
        # Create prompt for classification
        prompt = f"""
        Company Name: {company_name}
        Sector: {sector}
        Description: {company_description}
        
        Based on the company description above, is this company primarily 
        focused on artificial intelligence (AI) or machine learning (ML)?
        Please respond with either 'Yes' or 'No' and a brief explanation.
        """
        
        # Get classification from LLM
        response = await state.llm.ainvoke(prompt)
        
        # Store result in state
        if "results" not in state.data:
            state.data["results"] = []
            
        result = {
            "company_name": company_name,
            "sector": sector,
            "description": company_description,
            "classification": response
        }
        state.data["results"].append(result)
        
        # Update index for next company
        state.data["current_index"] = current_index + 1
        
        return state
    except Exception as e:
        print(f"Error in classify_company_node: {str(e)}")
        raise


async def print_results_node(state: ETLWorkflowState) -> ETLWorkflowState:
    """Print AI companies to console."""
    try:
        results = state.data.get("results", [])
        print("\nAI Companies Found:")
        print("==================")
        
        for company in results:
            if company.get("classification", "").lower() == "yes":
                print(f"\nName: {company.get('company_name', 'N/A')}")
                print(f"Sector: {company.get('sector', 'N/A')}")
                desc = company.get('description', 'N/A')
                print(f"Description: {desc}")
                print("---")
        
        return state
    except Exception as e:
        print(f"Error in print_results_node: {e}")
        raise


def should_continue(state: ETLWorkflowState) -> bool:
    """Determine if we should continue processing companies."""
    df = state.data["companies_df"]
    current_index = state.data["current_index"]
    return current_index < len(df)


async def main():
    """Set up and run the workflow."""
    # Initialize state
    state = ETLWorkflowState(llm=LLMManager())
    
    # Create workflow graph
    workflow = StateGraph(ETLWorkflowState)
    
    # Add nodes
    workflow.add_node("read", read_csv_file)
    workflow.add_node("classify", classify_company_node)
    workflow.add_node("print", print_results_node)
    
    # Add edges
    workflow.set_entry_point("read")
    workflow.add_edge("read", "classify")
    workflow.add_conditional_edges(
        "classify",
        should_continue,
        {
            True: "classify",
            False: "print"
        }
    )
    workflow.add_edge("print", END)
    
    # Compile workflow
    app = workflow.compile()
    
    # Run workflow
    final_state = await app.ainvoke(state)
    return final_state


if __name__ == "__main__":
    asyncio.run(main()) 