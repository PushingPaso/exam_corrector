"""
Example client that demonstrates how to use the MCP server with an LLM.
This shows the "agentic" approach where the LLM decides which tools to call.
"""

import asyncio
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import os


# Import MCP server
from exam.mcp import ExamMCPServer


class MCPClientDemo:
    """Demo client that uses MCP server tools with an LLM agent."""
    
    def __init__(self, exam_dir: Path = None):
        """Initialize the demo client."""
        self.mcp_server = ExamMCPServer(exam_dir)
        self.llm = ChatOpenAI(
            model="mistralai/mistral-7b-instruct",
            openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
        )
        self.langchain_tools = self._create_langchain_tools()
    
    def _create_langchain_tools(self):
        """Wrap MCP tools as LangChain tools."""
        
        langchain_tools = []
        
        # Wrap each MCP tool
        for tool_name, tool_func in self.mcp_server.tools.items():
            
            # Create a wrapper that preserves the original function signature
            # We need to handle different numbers of parameters
            
            if tool_name == "list_questions":
                @tool
                async def list_questions_tool() -> str:
                    """List all available questions in the question bank."""
                    return await self.mcp_server.tools["list_questions"]()
                langchain_tools.append(list_questions_tool)
            
            elif tool_name == "get_question":
                @tool
                async def get_question_tool(question_id: str) -> str:
                    """Get details of a specific question by ID."""
                    return await self.mcp_server.tools["get_question"](question_id)
                langchain_tools.append(get_question_tool)
            
            elif tool_name == "list_students":
                @tool
                async def list_students_tool(question_id: str = None) -> str:
                    """List all students who submitted answers, optionally filtered by question."""
                    return await self.mcp_server.tools["list_students"](question_id)
                langchain_tools.append(list_students_tool)
            
            elif tool_name == "read_student_answer":
                @tool
                async def read_student_answer_tool(question_id: str, student_code: str) -> str:
                    """Read a specific student's answer to a question."""
                    return await self.mcp_server.tools["read_student_answer"](question_id, student_code)
                langchain_tools.append(read_student_answer_tool)
            
            elif tool_name == "get_checklist":
                @tool
                async def get_checklist_tool(question_id: str) -> str:
                    """Get the assessment checklist for a specific question."""
                    return await self.mcp_server.tools["get_checklist"](question_id)
                langchain_tools.append(get_checklist_tool)
            
            elif tool_name == "assess_feature":
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
            
            elif tool_name == "calculate_score":
                @tool
                async def calculate_score_tool(assessments_json: str, max_score: float) -> str:
                    """Calculate the score based on feature assessments."""
                    return await self.mcp_server.tools["calculate_score"](assessments_json, max_score)
                langchain_tools.append(calculate_score_tool)
            
            elif tool_name == "search_course_material":
                @tool
                async def search_course_material_tool(query: str, max_results: int = 5) -> str:
                    """Search course materials for relevant content."""
                    return await self.mcp_server.tools["search_course_material"](query, max_results)
                langchain_tools.append(search_course_material_tool)
            
            elif tool_name == "generate_feedback":
                @tool
                async def generate_feedback_tool(assessments_json: str) -> str:
                    """Generate constructive feedback based on assessments."""
                    return await self.mcp_server.tools["generate_feedback"](assessments_json)
                langchain_tools.append(generate_feedback_tool)
            
            elif tool_name == "save_assessment":
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
    
    async def run_agent(self, task: str):
        """
        Run the agent with a given task.
        The agent will autonomously decide which tools to use.
        """
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent exam assessment assistant.
You have access to tools to:
- Read questions and student answers
- Get assessment checklists
- Evaluate features in answers
- Calculate scores
- Search course materials
- Generate feedback

Your job is to use these tools autonomously to complete assessment tasks.
Always be thorough and systematic in your evaluations."""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ])
        
        # Create agent
        agent = create_tool_calling_agent(self.llm, self.langchain_tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.langchain_tools,
            verbose=True,
            max_iterations=15,
            return_intermediate_steps=True
        )
        
        # Run agent
        print("\n" + "="*70)
        print(f"TASK: {task}")
        print("="*70 + "\n")
        
        result = await agent_executor.ainvoke({"input": task})
        
        print("\n" + "="*70)
        print("AGENT RESULT:")
        print("="*70)
        print(result["output"])
        print("\n")
        
        return result


async def demo_simple_assessment():
    """Demo 1: Simple assessment of one student."""
    print("\n DEMO 1: Simple Student Assessment")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"))
    
    task = """
    Assess the answer of student with one of the codes you find to question BuildAutomation-1.
    Use the checklist and evaluate all SHOULD features.
    Calculate the final score and provide feedback.
    """
    
    await client.run_agent(task)


async def demo_batch_processing():
    """Demo 2: Process multiple students automatically."""
    print("\n DEMO 2: Batch Processing")
    print("="*70)
    
    client = MCPClientDemo(Path("mock_exam_submissions"))
    
    task = """
    Process all students for question Foundations-1:
    1. List all students who answered this question
    2. For the first 3 students, assess their answers and calculate scores
    3. Generate a summary showing each student's score
    """
    
    await client.run_agent(task)


async def main():
    """Run demos."""
    print("\n" + "="*70)
    print("MCP CLIENT DEMO - Agentic Exam Assessment")
    print("="*70)
    
    # Check if we have mock exam data
    if not Path("mock_exam_submissions").exists():
        print("\n  No mock exam data found!")
        print("Run: python generate_mock_exam.py")
        return
    
    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n  OPENROUTER_API_KEY not set!")
        print("Set it with: export OPENROUTER_API_KEY='your-key'")
        return
    
    # Run demos
    demos = [
        ("Simple Assessment", demo_simple_assessment),
        ("Batch Processing", demo_batch_processing),
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    print("0. Run all demos")
    
    choice = input("\nSelect demo (0-2): ").strip()
    
    if choice == "0":
        for name, demo_func in demos:
            await demo_func()
            input("\nPress Enter to continue to next demo...")
    elif choice in ["1", "2"]:
        idx = int(choice) - 1
        await demos[idx][1]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())