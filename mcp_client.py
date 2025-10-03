"""
MCP Client using Groq 
"""

import asyncio
import json
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import os

# Import MCP server
from exam.mcp import ExamMCPServer


class MCPClientDemo:
    """Client using Groq's"""
    
    def __init__(self, exam_dir: Path = None, model: str = "llama-3.3"):

        self.mcp_server = ExamMCPServer(exam_dir)
        
        # Check API key
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError(
                "GROQ_API_KEY not set!\n"
                "Get a free key at: https://console.groq.com/keys\n"
                "Then: export GROQ_API_KEY='gsk_...'"
            )
        
        # Model configurations
        model_configs = {
            "llama-3.3": {
                "name": "llama-3.3-70b-versatile",
                "display": "Llama 3.3 70B Versatile (BEST!)",
                "max_tokens": 8000
            },
            "llama-3.1": {
                "name": "llama-3.1-70b-versatile",
                "display": "Llama 3.1 70B Versatile",
                "max_tokens": 8000
            },
            "mixtral": {
                "name": "mixtral-8x7b-32768",
                "display": "Mixtral 8x7B",
                "max_tokens": 8000
            },
            "llama-8b": {
                "name": "llama-3.1-8b-instant",
                "display": "Llama 3.1 8B Instant (Fastest)",
                "max_tokens": 8000
            }
        }
        
        if model not in model_configs:
            print(f"  Unknown model '{model}', using llama-3.3")
            model = "llama-3.3"
        
        config = model_configs[model]
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model=config["name"],
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=config["max_tokens"],
        )
        
        self.langchain_tools = self._create_langchain_tools()
    
    def _create_langchain_tools(self):
        """Wrap MCP tools as LangChain tools."""
        
        langchain_tools = []
        
        @tool
        async def list_questions_tool() -> str:
            """List all available questions in the question bank."""
            return await self.mcp_server.tools["list_questions"]()
        langchain_tools.append(list_questions_tool)
        
        @tool
        async def get_question_tool(question_id: str) -> str:
            """Get details of a specific question by ID."""
            return await self.mcp_server.tools["get_question"](question_id)
        langchain_tools.append(get_question_tool)
        
        @tool
        async def list_students_tool(question_id: str = None) -> str:
            """List all students who submitted answers, optionally filtered by question."""
            return await self.mcp_server.tools["list_students"](question_id)
        langchain_tools.append(list_students_tool)
        
        @tool
        async def read_student_answer_tool(question_id: str, student_code: str) -> str:
            """Read a specific student's answer to a question."""
            return await self.mcp_server.tools["read_student_answer"](question_id, student_code)
        langchain_tools.append(read_student_answer_tool)
        
        @tool
        async def get_checklist_tool(question_id: str) -> str:
            """Get the assessment checklist for a specific question."""
            return await self.mcp_server.tools["get_checklist"](question_id)
        langchain_tools.append(get_checklist_tool)
        
        @tool
        async def assess_feature_tool(
            question_id: str,
            feature_description: str,
            feature_type: str,
            student_answer: str
        ) -> str:
            """Assess whether a feature is present in a student's answer."""
            return await self.mcp_server.tools["assess_feature"](
                question_id, feature_description, feature_type, student_answer
            )
        langchain_tools.append(assess_feature_tool)
        
        @tool
        async def calculate_score_tool(assessments_json: str, max_score: float) -> str:
            """Calculate the score based on feature assessments."""
            return await self.mcp_server.tools["calculate_score"](assessments_json, max_score)
        langchain_tools.append(calculate_score_tool)
        
        @tool
        async def search_course_material_tool(query: str, max_results: int = 5) -> str:
            """Search course materials for relevant content."""
            return await self.mcp_server.tools["search_course_material"](query, max_results)
        langchain_tools.append(search_course_material_tool)
        
        @tool
        async def generate_feedback_tool(assessments_json: str) -> str:
            """Generate constructive feedback based on assessments."""
            return await self.mcp_server.tools["generate_feedback"](assessments_json)
        langchain_tools.append(generate_feedback_tool)
        
        @tool
        async def save_assessment_tool(
            question_id: str,
            student_code: str,
            score: float,
            assessments_json: str,
            feedback: str = ""
        ) -> str:
            """Save assessment results for a student."""
            return await self.mcp_server.tools["save_assessment"](
                question_id, student_code, score, assessments_json, feedback
            )
        langchain_tools.append(save_assessment_tool)
        
        return langchain_tools
    
    async def run_agent(self, task: str, verbose: bool = True):
        """Run the agent with a given task."""
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an exam assessment assistant with access to tools.

IMPORTANT: When you need to use a tool, you MUST call it properly using the tool calling mechanism.
Do NOT generate XML or text descriptions of tool calls - actually invoke the tools.

Your capabilities:
- Read questions and student answers
- Access assessment checklists
- Evaluate features in answers
- Calculate scores
- Search course materials
- Generate feedback

Be systematic and thorough. Call tools one at a time and wait for results."""),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}"),
])
        
        # Create agent
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
        
        # Run agent
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
            print(f"\n  Completed in {elapsed:.2f} seconds")
            print("\n")
            
            return result
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()
            return None


async def demo_simple():
    """Demo 1: Simple assessment."""
    print("\n DEMO 1: Simple Student Assessment")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"), model="llama-3.3")
    
    await client.run_agent("""
        Find the student with code 280944 who answered "BuildAutomation-2".
        Assess their answer using the checklist.
        Calculate score and provide brief feedback.
    """)


async def demo_compare():
    """Demo 2: Compare students."""
    print("\n DEMO 2: Compare Students")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"), model="llama-3.3")
    
    await client.run_agent("""
        Find the first 2 students who answered Foundations-1.
        Assess both and tell me who did better and why.
    """)


async def demo_rag():
    """Demo 3: RAG-enhanced."""
    print("\n DEMO 3: RAG-Enhanced Assessment")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"), model="llama-3.3")
    
    await client.run_agent("""
        Before assessing a student on Foundations-2:
        1. Search course materials for "algorithm"
        2. Use that context to assess the student's answer
        3. In feedback, cite specific course materials to review
    """)


async def main():
    
    # Check prerequisites
    if not Path("mock_exam_submissions").exists():
        print("\n  No mock exam data!")
        print("Run: python generate_mock_exam.py")
        return
    
    if not os.environ.get("GROQ_API_KEY"):
        print("\n  GROQ_API_KEY not set!")
        print("\n1. Go to: https://console.groq.com/keys")
        print("2. Sign up (free)")
        print("3. Create API key")
        print("4. export GROQ_API_KEY='gsk_...'")
        return
    
    print("\n API Key configured")
    
    # Demos
    demos = [
        ("Simple Assessment", demo_simple),
        ("Compare Students", demo_compare),
        ("RAG-Enhanced", demo_rag),
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    print("0. Run all")
    
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