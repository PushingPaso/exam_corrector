"""
MCP Client con sistema di tool collaborativi.
"""

import asyncio
import json
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import os

# Import MCP server aggiornato
from exam.mcp import ExamMCPServer


class MCPClientDemo:
    """Client con sistema di tool collaborativi."""
    
    def __init__(self, exam_dir: Path = None, model: str = "llama-3.3"):
        self.mcp_server = ExamMCPServer(exam_dir)
        
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not set!")
        
        model_configs = {
            "llama-3.3": "llama-3.3-70b-versatile",
            "llama-3.1": "llama-3.1-70b-versatile",
            "mixtral": "mixtral-8x7b-32768",
            "llama-8b": "llama-3.1-8b-instant"
        }
        
        model_name = model_configs.get(model, model_configs["llama-3.3"])
        
        self.llm = ChatGroq(
            model=model_name,
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=8000,
        )
        
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
        
        # BATCH TOOLS
        
        @tool
        async def list_available_exams_tool() -> str:
            """
            List all available exam YAML files in static/se-exams directory.
            
            Use this to discover which exams are available for correction.
            Shows exam pairs (questions + responses files) and their status.
            """
            return await self.mcp_server.tools["list_available_exams"]()
        langchain_tools.append(list_available_exams_tool)
        
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
- list_questions: See all questions
- list_students: See all students
- load_student_answer: Load an answer into memory
- load_checklist: Load grading criteria into memory
- calculate_score: Get final score (needs assessments done)

COMPOSED TOOLS (quick workflows):
- assess_all_features: Complete assessment in one step (loads everything, assesses all, calculates score)

BATCH EXAM TOOLS (for entire exams from static/se-exams/):
- list_available_exams: See which exam YAML files are available
- load_exam_from_yaml: Load questions and responses from YAML files
- assess_student_exam: Assess all responses for one student from loaded exam
- assess_full_exam: AUTOMATIC CORRECTION - assess entire exam for all students

IMPORTANT: Exam YAML files are stored in static/se-exams/ directory.
Just use filenames like "se-2025-06-05-questions.yml", not full paths.

WORKFLOW STRATEGIES:

For single answer assessment:
  Example: "Assess student 280944 on CI-5"
  -> assess_all_features("CI-5", "280944")

For comparing students:
  Example: "Compare students on CI-5"
  -> load_checklist("CI-5")
  -> assess_all_features("CI-5", "280944")
  -> assess_all_features("CI-5", "280945")

For batch exam correction (AUTOMATIC):
  Example: "What exams are available? Correct the June 2025 exam"
  -> list_available_exams()
  -> assess_full_exam("se-2025-06-05-questions.yml", "se-2025-06-05-responses.yml")

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
    """Demo 1: Quick assessment using composed tool."""
    print("\nDEMO 1: Quick Assessment (Composed Tool)")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"), model="llama-3.3")
    
    await client.run_agent("""
        Assess student 280944's answer to question CI-5.
        Use the fastest method.
    """)


async def demo_compare():
    """Demo 2: Compare students using atomic tools."""
    print("\nDEMO 2: Compare Students (Atomic Tools)")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"), model="llama-3.3")
    
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
    
    client = MCPClientDemo(Path("mock_exam_submissions"), model="llama-3.3")
    
    await client.run_agent("""
        Assess ALL students who answered CI-5:
        1. Find all students
        2. Load checklist once
        3. Assess each student completely
        4. Create a ranking by score
        5. Show me the top 3
    """)


async def demo_full_exam():
    """Demo 4: Automatic exam correction from YAML files."""
    print("\nDEMO 4: Automatic Exam Correction")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"), model="llama-3.3")
    
    await client.run_agent("""
        Correct the June 2025 exam automatically:
        1. First, list available exams to see what we have
        2. Load se-2025-06-05-questions.yml and se-2025-06-05-responses.yml
        3. Assess only the first 2 students (for testing)
        4. Show me statistics and individual results
    """)


async def demo_student_exam():
    """Demo 5: Assess full exam for one student."""
    print("\nDEMO 5: Single Student Full Exam Assessment")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"), model="llama-3.3")
    
    await client.run_agent("""
        Assess one student's entire exam:
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
    
    print("\nReady to run demos!")
    
    demos = [
        ("Quick Assessment", demo_simple),
        ("Compare Students", demo_compare),
        ("Batch Assessment", demo_batch),
        ("Automatic Exam Correction", demo_full_exam),
        ("Single Student Full Exam", demo_student_exam),
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  0. Run all")
    
    choice = input("\nSelect (0-5): ").strip()
    
    try:
        if choice == "0":
            for name, func in demos:
                await func()
                input("\n[Press Enter for next demo...]")
        elif choice in ["1", "2", "3", "4", "5"]:
            await demos[int(choice)-1][1]()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n\nInterrupted")


if __name__ == "__main__":
    asyncio.run(main())