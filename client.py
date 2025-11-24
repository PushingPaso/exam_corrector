from langchain.agents import create_agent

from exam.llm_provider import llm_client
from exam.mcp import ExamMCPServer


def main():
    exam_date = input("Insert date exam: ")
    mcp = ExamMCPServer()
    llm,_,_ = llm_client()
    agent = create_agent(
        llm,
        tools=[mcp.assess_student_exam,mcp.load_exam_from_yaml_tool,mcp.load_checklist]
    )
    prompt = f"using the tool asses the exam of the {exam_date}: First load the exam file from static/se_exam/se-{exam_date}-grades.yaml, se-{exam_date}-question.yaml, se-{exam_date}-resposes.yaml than upload the checklist and evaluate the relative student IMPORTANT: USE THE TOOL"
    agent.invoke(prompt)

if __name__ == '__main__':
    main()