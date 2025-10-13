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
                
        
        @tool
        async def load_exam_from_yaml_tool(questions_file: str, responses_file: str) -> str:
            """
            Load an entire exam from YAML files in static/se-exams directory.
            
            Args:
                questions_file: Filename only (e.g., "se-2025-06-05-questions.yml")
                responses_file: Filename only (e.g., "se-2025-06-05-responses.yml")
            
            Files are automatically searched in static/se-exams/ directory.
            Use list_available_exams to see available files first.
            """
            return await self.mcp_server.tools["load_exam_from_yaml"](questions_file, responses_file)
        langchain_tools.append(load_exam_from_yaml_tool)
        
        @tool
        async def assess_student_exam_tool(student_email: str) -> str:
            """
            Assess all responses for a single student from a loaded exam.
            Requires load_exam_from_yaml to be called first.
            
            Args:
                student_email: Student's email (can use first 20 characters)
            
            Returns complete assessment of all the student's exam responses.
            """
            return await self.mcp_server.tools["assess_student_exam"](student_email)
        langchain_tools.append(assess_student_exam_tool)
        
        @tool
        async def assess_full_exam_tool(questions_file: str, responses_file: str, max_students: int = None) -> str:
            """
            Assess an ENTIRE exam for ALL students automatically.
            
            This is the COMPLETE automation tool - loads the exam and assesses everyone.
            
            Args:
                questions_file: Path to questions YAML
                responses_file: Path to responses YAML
                max_students: Optional limit for testing (e.g., 3 for first 3 students)
            
            Returns summary with statistics and results for all students.
            Perfect for automatic exam correction.
            """
            return await self.mcp_server.tools["assess_full_exam"](questions_file, responses_file, max_students)
        langchain_tools.append(assess_full_exam_tool)
        
        return langchain_tools
    
    async def run_agent(self, task: str, verbose: bool = True):
        """Run the agent with a given task."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an exam assessment assistant with ATOMIC and COMPOSED tools.
             
            IMPORTANT: When you need to use a tool, you MUST call it properly using the tool calling mechanism.
            Do NOT generate XML or text descriptions of tool calls - actually invoke the tools.

            ATOMIC TOOLS (building blocks):

            - load_student_answer: Load an answer into memory
            - load_checklist: Load grading criteria into memory
            - assess_all_features:
            - load_exam_from_yaml_tool: load the exam from yaml file
            - evaluate a full exam:
            - assess_student_exam_tool:evaluate an axam for a single student
            - assess_full_exam_tool: evaluate exam for multiple students


            IMPORTANT: Exam YAML files are stored in static/se-exams/ directory.
            Just use filenames like "se-2025-06-05-questions.yml", not full paths.

            WORKFLOW STRATEGIES:

            For single answer assessment:
            Example: "Assess student 280944 on CI-5"
            -> assess_all_features("CI-5", "280944")


            For single student full exam:
            Example: "Assess all responses for first student in June 2025 exam"
            -> load_exam_from_yaml("se-2025-06-05-questions.yml", "se-2025-06-05-responses.yml")
            -> assess_student_exam("1377db8e05e4...")

            Be systematic. Choose the right tool for the task. Call tools one at a time and wait for results"""),
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
    """Quick assessment using composed tool."""
    print("\nDEMO 1: Quick Assessment (Composed Tool)")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"))
    
    await client.run_agent("""
        Assess and evaluate student 280944's answer to question DistributedSystems-30.
    """)

async def demo_full_exam():
    """Automatic exam correction from YAML files."""
    print("\nDEMO 4: Automatic Exam Correction")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"))
    
    await client.run_agent("""
        Correct the June 2025 exam automatically:
        1. First, list available exams to see what we have
        2. Load se-2025-06-05-questions.yml and se-2025-06-05-responses.yml
        3. Assess only the first 2 students (for testing)
        4. Show me statistics and individual results
    """)


async def demo_student_exam():
    """Assess full exam for one student."""
    print("\nDEMO 5: Single Student Full Exam Assessment")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"))
    
    await client.run_agent("""
        Load the se-2025-06-05 exam files, find the first student and assess all his 9 responses
        show me his total score and which questions they did well/poorly on.
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
    
    print("\nReady to run exam corrector")
    
    demos = [
        ("Evaluate one answear from a student", demo_simple),
        ("Full Exam Correction", demo_full_exam),
        ("Single Student Exam Correction", demo_student_exam),
    ]
    
    print("\nAvailable function:")
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