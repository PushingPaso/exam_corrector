"""
MCP Client con sistema di tool collaborativi.
"""

import asyncio
import json
from pathlib import Path
from exam.llm_provider import llm_client
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import os

# Import MCP server aggiornato
from exam.mcp import ExamMCPServer


class MCPClientDemo:
    """Client con sistema di tool collaborativi."""
    
    def __init__(self, exam_dir: Path = None):
        self.mcp_server = ExamMCPServer(exam_dir)
        
        self.llm,_,_ = llm_client()
        
        self.langchain_tools = self._create_langchain_tools()
    
    def _create_langchain_tools(self):
        """Crea i wrapper LangChain per tutti i tool."""
        
        langchain_tools = []
        
        @tool
        async def list_questions_tool() -> str:
            """List all available questions in the question bank."""
            return await self.mcp_server.tools["list_questions"]()
        langchain_tools.append(list_questions_tool)
        
        @tool
        async def list_students_tool(question_id: str = None) -> str:
            """List all students who submitted answers, optionally filtered by question."""
            return await self.mcp_server.tools["list_students"](question_id)
        langchain_tools.append(list_students_tool)
        
        @tool
        async def load_student_answer_tool(question_id: str, student_code: str) -> str:
            """
            Load a student's answer into memory.
            The answer will be available for other tools to use.
            
            Use this when you need to:
            - Read an answer before assessing it
            - Compare multiple answers (load each one)
            - Prepare for assessment
            """
            return await self.mcp_server.tools["load_student_answer"](question_id, student_code)
        langchain_tools.append(load_student_answer_tool)
        
        @tool
        async def load_checklist_tool(question_id: str) -> str:
            """
            Load the assessment checklist for a question into memory.
            The checklist will be available for other tools to use.
            
            Use this when you need to:
            - See what features will be assessed
            - Prepare for assessing multiple students on the same question
            - Understand the grading criteria
            """
            return await self.mcp_server.tools["load_checklist"](question_id)
        langchain_tools.append(load_checklist_tool)
        
        @tool
        async def calculate_score_tool(question_id: str, student_code: str) -> str:
            """
            Calculate final score from stored assessments.
            Requires that assess_all_features has been run first.
            
            Use this to:
            - Get the final score
            - See the breakdown by feature type
            - Understand how the score was calculated
            """
            return await self.mcp_server.tools["calculate_score"](question_id, student_code)
        langchain_tools.append(calculate_score_tool)
        
        @tool
        async def assess_all_features_tool(question_id: str, student_code: str) -> str:
            """
            Assess ALL features for a student's answer in one step.
            Automatically loads answer and checklist if needed.
            
            This is the FASTEST way to get a complete assessment.
            
            Use this when you need:
            - Quick complete assessment
            - Full score breakdown
            - All feature evaluations at once
            
            Returns: Complete assessment with score and all feature feedback.
            """
            return await self.mcp_server.tools["assess_all_features"](question_id, student_code)
        langchain_tools.append(assess_all_features_tool)
        
        return langchain_tools
    
    async def run_agent(self, task: str, verbose: bool = True):
        """Run the agent with a given task."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an exam assessment assistant with ATOMIC and COMPOSED tools.
             
             IMPORTANT: When you need to use a tool, you MUST call it properly using the tool calling mechanism.
Do NOT generate XML or text descriptions of tool calls - actually invoke the tools.

ATOMIC TOOLS (building blocks):
- list_questions: See all questions
- list_students: See all students
- load_student_answer: Load an answer into memory
- load_checklist: Load grading criteria into memory
- calculate_score: Get final score (needs assessments done)

COMPOSED TOOLS:
- assess_all_features: Complete assessment in one step (assesses all, calculates score)

WORKFLOW STRATEGIES:

             
  Example: "Assess student 280944 on CI-5"

-> load_student_answer("CI-5", "280944")
-> load_checklist("CI-5")
-> assess_all_features("CI-5", "280944")

For comparing students: Load checklist once, then assess each
  Example: "Compare students on CI-5"
  -> load_checklist("CI-5")
  -> assess_all_features("CI-5", "280944")
  -> assess_all_features("CI-5", "280945")

For batch assessment: Load checklist once, assess all students
  Example: "Assess all students on CI-5"
  -> load_checklist("CI-5")
  -> list_students("CI-5")
  -> assess_all_features for each student

Be systematic. Choose the right tool for the task.Call tools one at a time and wait for results"""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.langchain_tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.langchain_tools,
            verbose=verbose,
            max_iterations=20,
            max_execution_time=180,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        print("\n" + "="*70)
        print(f"TASK: {task}")
        print("="*70 + "\n")
        
        import time
        start = time.time()
        
        try:
            result = await agent_executor.ainvoke({"input": task})
            elapsed = time.time() - start
            
            print("\n" + "="*70)
            print("RESULT:")
            print("="*70)
            print(result["output"])
            print(f"\nCompleted in {elapsed:.2f} seconds")
            print("="*70 + "\n")
            
            return result
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return None


async def demo_simple():
    """Demo 1: Quick assessment using composed tool."""
    print("\nDEMO 1: Quick Assessment (Composed Tool)")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"))
    
    await client.run_agent("""
       
       Find the student with code 280944 who answered  at thre question with id "CI-5".
        Get the checklist, use it to assess the answer and save the feedback.
    """)


async def demo_compare():
    """Demo 2: Compare students using atomic tools."""
    print("\nDEMO 2: Compare Students (Atomic Tools)")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"))
    
    await client.run_agent("""
        Compare the first 2 students who answered CI-5:
        1. Load the checklist once (reuse it)
        2. Assess both students
        3. Tell me who scored better and why
        4. Focus especially on CORE features
    """)


async def demo_batch():
    """Demo 3: Batch assessment."""
    print("\nDEMO 3: Batch Assessment")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"))
    
    await client.run_agent("""
        Assess ALL students who answered CI-5:
        1. Find all students
        2. Load checklist once
        3. Assess each student completely
        4. Create a ranking by score
        5. Show me the top 3
    """)


async def main():
    """Main menu."""
    
    if not Path("mock_exam_submissions").exists():
        print("\nNo mock exam data! Run: python generate_mock_exam.py")
        return
    
    if not os.environ.get("GROQ_API_KEY"):
        print("\nGROQ_API_KEY not set!")
        print("Get free key at: https://console.groq.com/keys")
        return
    
    print("\nReady to run demos!")
    
    demos = [
        ("Quick Assessment", demo_simple),
        ("Compare Students", demo_compare),
        ("Batch Assessment", demo_batch),
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  0. Run all")
    
    choice = input("\nSelect (0-3): ").strip()
    
    try:
        if choice == "0":
            for name, func in demos:
                await func()
                input("\n[Press Enter for next demo...]")
        elif choice in ["1", "2", "3"]:
            await demos[int(choice)-1][1]()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n\nInterrupted")


if __name__ == "__main__":
    asyncio.run(main())