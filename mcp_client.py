"""
MCP Client con sistema di tool collaborativi.
"""

import asyncio
import os
from pathlib import Path
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from exam.llm_provider import llm_client
# Import MCP server aggiornato
from exam.mcp import ExamMCPServer


class MCPClientDemo:
    """Client con sistema di tool collaborativi."""
    
    def __init__(self):
        self.mcp_server = ExamMCPServer()
        self.llm,_,_ = llm_client()
        self.langchain_tools = self._create_langchain_tools()

    
    def _create_langchain_tools(self):
        """Crea i wrapper LangChain per tutti i tool."""
        
        langchain_tools = []
        
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
        
        return langchain_tools
    
    async def run_agent(self, task: str, verbose: bool = True):
        """Run the agent with a given task."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an exam assessment assistant with ATOMIC and COMPOSED tools.
             
            IMPORTANT: When you need to use a tool, you MUST call it properly using the tool calling mechanism.
            Do NOT generate XML or text descriptions of tool calls - actually invoke the tools.

            ATOMIC TOOLS (building blocks):

            - load_checklist: Load grading criteria into memory
            - load_exam_from_yaml_tool: load the exam from yaml file
            - assess_student_exam_tool:evaluate an axam for a single student


            IMPORTANT: Exam YAML files are stored in static/se-exams/ directory.
            Just use filenames like "se-2025-06-05-questions.yml", not full paths.

            For the correct fuctioning of the program you must do ths
            -> load_checklist: Load grading criteria into memory
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


async def demo_student_exam():
    """Assess full exam for one student."""
    print("\nDEMO 5: Single Student Full Exam Assessment")
    print("="*70)
    
    client = MCPClientDemo()
    
    await client.run_agent("""
        First you need to Load the se-2025-06-05 exam question and response, 
        for the first student assess all his 9 answer of the exam saving the result in the evaluation package
    """)


async def main():
    """Main menu."""
    
    if not os.environ.get("GROQ_API_KEY"):
        print("\nGROQ_API_KEY not set!")
        print("Get free key at: https://console.groq.com/keys")
        return
    
    print("\nReady to run exam corrector")

    await demo_student_exam()



if __name__ == "__main__":
    asyncio.run(main())