from langchain.agents import create_agent
from exam.llm_provider import llm_client
from exam.mcp import ExamMCPServer
import mlflow


def main():
    mlflow.set_tracking_uri("http://localhost:5000")

    exam_date = input("Insert date exam: ")
    mcp = ExamMCPServer()
    llm,_,_ = llm_client()
    agent = create_agent(
        llm,
        tools=[mcp.assess_student_exam,mcp.load_exam_from_yaml_tool,mcp.load_checklist]
    )
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(agent, name="langchain_model")

    # Load the logged model using MLflow's Python function flavor
    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

    # Predict using the loaded model
    print(loaded_model.predict([{"input": f"using the tool asses the exam of the {exam_date}: First load the exam file from static/se_exam/se-{exam_date}-grades.yaml, se-{exam_date}-question.yaml, se-{exam_date}-resposes.yaml than upload the checklist and evaluate the relative student"
                 f"IMPORTANT: USE THE TOOL" }]))

if __name__ == '__main__':
    main()