import getpass
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_models import ChatOllama 


load_dotenv()
import os
os.environ["GOOGLE_API_KEY"] = "your_gemini_api_key"



llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # or "gemini-1.5-pro" if you have access
    temperature=0.7
)

# Create tools for the research agent
search = TavilySearchResults()
tools = [search]

# Define agent states
class AgenticSystem(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], lambda x, y: x + y] # Conversation history
    question: str # Original user query
    research_data: str # Collected data
# Research Agent
def create_research_agent(llm, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research agent. """),
        ("placeholder", "{agent_scratchpad}"),
        ("user", "{input}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

research_agent = create_research_agent(llm, tools)

def research_node(state: AgenticSystem):
    print("\nResearch agent working...")
    response = research_agent.invoke({"input": state["question"]})
    research_data = response["output"]
    return {"messages": [AIMessage(content=f"Research data: {research_data}")], "research_data": research_data}

# Answer Drafting Agent
def create_drafting_agent(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an answer drafting agent."""),
        ("user", """Original question: {question}
        
        Research data: {research_data}
        
        Please compose the final answer."""),
    ])
    return prompt | llm

drafting_agent = create_drafting_agent(llm)

def drafting_node(state: AgenticSystem):
    print("\nDrafting agent working...")
    response = drafting_agent.invoke({
        "question": state["question"],
        "research_data": state["research_data"]
    })
    return {"messages": [AIMessage(content=response.content)]}

# Create the workflow
workflow = StateGraph(AgenticSystem)


# Add nodes
workflow.add_node("research", research_node)
workflow.add_node("draft", drafting_node)

# Set edges
workflow.add_edge("research", "draft")
workflow.add_edge("draft", END)


# Set entry point
workflow.set_entry_point("research")

# Compile the graph
app = workflow.compile()

# Run the workflow
def run_dual_agent(question):
    print(f"\nProcessing question: {question}")
    result = app.invoke({
        "messages": [],
        "question": question,
        "research_data": ""
    })
    
    # Extract and return the final answer
    final_message = result["messages"][-1].content
    return final_message

# Example usage
if __name__ == "__main__":
    question = input("Search your Query:")
    answer = run_dual_agent(question)
    print("\nFinal Answer:")
    print(answer)











