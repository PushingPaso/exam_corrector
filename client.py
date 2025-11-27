"""
Client Agente Singolo con integrazione MLflow.
Replica la logica di multiAgents_client utilizzando LangChain AgentExecutor standard.
"""

import asyncio

from langchain_community.callbacks import get_openai_callback
from langchain_core.callbacks import StdOutCallbackHandler, BaseCallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool

from exam.llm_provider import llm_client
from exam.mcp import ExamMCPServer

async def main():

    console_handler = StdOutCallbackHandler()

    my_callbacks = [console_handler]

    print("\n Multi-Agent Exam Assessor")
    print("=" * 70)
    exam_date = input("\nExam date (format YYYY-MM-DD, e.g., 2025-06-05): ").strip()
    if not exam_date:
        exam_date = "2025-06-05"

    mcp = ExamMCPServer()
    llm, model_name, _ = llm_client()

    upload_prompt = (
        "You are a technical agent responsible ONLY for data loading (ETL). "
        "1. Load the exam from the YAML files. "
        "2. Use the question IDs to load the checklists. "
        "3. IMPORTANT: Once everything is loaded, DO NOT ask for user input. "
        "Simply reply: 'DATA LOADED SUCCESSFULLY. EXAM ID: [id]. READY FOR ASSESSMENT'. "
        "Be brief and concise."
    )

    assess_prompt = (
        "You are an assessment agent. The data is already loaded into the shared context (ExamMCPServer). "
        "1. FIRST, call the 'list_loaded_students_tool' tool to get the list of student emails. "
        "2. THEN, list and call 'assess_students_batch' "
    )

    print(f"\n[CLIENT] Starting assessment for exam date {exam_date}...\n")

    uploaderAgent = create_agent(
        llm,
        tools=[mcp.load_exam_from_yaml_tool, mcp.load_checklist],
        system_prompt=upload_prompt
    )

    assessorAgent = create_agent(
        llm,
        tools=[mcp.assess_students_batch, mcp.list_students], # Assicurati che in mcp/__init__.py siano staticmethod e decorati corretti
        system_prompt=assess_prompt
    )

    @tool(
        "uploaderAgent",
        description="MANDATORY FIRST STEP. Call this agent to load exam data. Input: string description."
    )
    async def call_uploaderAgent(upload_query: str):
        print(f"\n--- [Supervisor] Call to Uploader Agent ---\n")
        # 3. PASSA I CALLBACK NELLA CONFIG
        result = await uploaderAgent.ainvoke(
            {"messages": [{"role": "user", "content": upload_query}]},
            config={"callbacks": my_callbacks}
        )
        return result["messages"][-1].content

    @tool(
        "assessorAgent",
        description="MANDATORY SECOND STEP. Call this agent ONLY AFTER uploaderAgent finishes. Input: 'Assess all students'."
    )
    async def call_assessorAgent(query: str):
        print(f"\n--- [Supervisor] Call to Assessor Agent ---\n")
        # 3. PASSA I CALLBACK NELLA CONFIG
        result = await assessorAgent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"callbacks": my_callbacks}
        )
        return result["messages"][-1].content

    supervisor = create_agent(llm, tools=[call_uploaderAgent, call_assessorAgent])

    supervisor_query = (
        f"Execute the assessment pipeline for exam date {exam_date}.\n"
        f"Files: se-{exam_date}-questions.yml, se-{exam_date}-responses.yml, se-{exam_date}-grades.yml.\n\n"
        "PLAN:\n"
        "1. Call uploaderAgent to load data. Wait for 'DATA LOADED'.\n"
        "2. IMMEDIATELY Call assessorAgent to evaluate students.\n"
        "IMPORTANT: Do NOT call both agents simultaneously."
    )

    print("\n--- [System] Starting Supervisor Process ---\n")

    with get_openai_callback() as cb:
        await supervisor.ainvoke(
            {"messages": [{"role": "user", "content": supervisor_query}]},
            config={"callbacks": my_callbacks}
        )
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost:.4f}")


if __name__ == '__main__':
    asyncio.run(main())