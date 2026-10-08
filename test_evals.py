import os
import sys
from langsmith import Client
from langsmith.evaluation import evaluate
from app.rag_pipeline import get_rag_chain

def check_answer_presence(run, example):
    response = run.outputs.get("answer", "")
    score = 1 if len(response.strip()) > 10 else 0
    return {"score": score, "key": "answer_length_check"}

def run_qa_evaluation():
    client = Client()
    dataset_name = "Corporate_RAG_QA_Dataset"
    
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name, description="Golden QA tests")
        client.create_examples(
            inputs=[{"input": "What days do we have to work from the office?"}],
            outputs=[{"output": "Tuesdays and Thursdays"}],
            dataset_id=dataset.id
        )
    
    def target(inputs):
        chain = get_rag_chain()
        return chain.invoke({"input": inputs["input"]})

    print("🚀 Starting automated LangSmith validation evaluation...")
    evaluate(target, data=dataset_name, evaluators=[check_answer_presence], experiment_prefix="ci-cd-validation")

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    run_qa_evaluation()