"""
Simple MCP demo WITHOUT agent framework.
This demonstrates MCP tools being called directly in a workflow.
More reliable than agent-based approach.
"""

import asyncio
import json
from pathlib import Path
from exam.mcp import ExamMCPServer


async def demo_simple_workflow():
    """
    Demo: Simple assessment workflow using MCP tools directly.
    This is more reliable than using an agent.
    """
    print("\n" + "="*70)
    print("SIMPLE MCP WORKFLOW DEMO")
    print("="*70)
    
    # Initialize MCP server
    exam_dir = Path("mock_exam_submissions")
    if not exam_dir.exists():
        print(" No mock exam data found!")
        print("Run: python generate_mock_exam.py")
        return
    
    server = ExamMCPServer(exam_dir)
    tools = server.tools
    
    print("\n MCP Server initialized")
    print(f" {len(tools)} tools available\n")
    
    # Step 1: List available questions
    print("Step 1: Listing available questions...")
    questions_json = await tools["list_questions"]()
    questions = json.loads(questions_json)
    print(f"✓ Found {len(questions)} questions")
    
    # Pick first question
    question = questions[383]
    question_id = question["id"]
    question_text = question["text"]
    max_score = question["weight"]
    
    print(f"✓ Selected: {question_id} - {question_text}")
    print(f"✓ Max score: {max_score}\n")
    
    # Step 2: List students for this question
    print("Step 2: Finding students...")
    students_json = await tools["list_students"](question_id)
    students = json.loads(students_json)
    
    if not students:
        print(" No students found for this question")
        return
    
    print(f" Found {len(students)} students")
    
    # Pick first student
    student = students[0]
    student_code = student["code"]
    student_name = student["name"]
    print(f" Selected: {student_name} ({student_code})\n")
    
    # Step 3: Read student's answer
    print("Step 3: Reading student answer...")
    answer_json = await tools["read_student_answer"](question_id, student_code)
    answer_data = json.loads(answer_json)
    answer_text = answer_data["answer"]
    word_count = answer_data["word_count"]
    
    print(f" Answer length: {word_count} words")
    print(f" Preview: {answer_text[:100]}...\n")
    
    # Step 4: Get assessment checklist
    print("Step 4: Loading assessment checklist...")
    checklist_json = await tools["get_checklist"](question_id)
    checklist = json.loads(checklist_json)
    
    should_features = checklist.get("should", [])
    shouldnt_features = checklist.get("should_not", [])
    
    print(f" {len(should_features)} SHOULD features")
    print(f" {len(shouldnt_features)} SHOULDNT features\n")
    
    # Step 5: Assess each feature
    print("Step 5: Assessing features...")
    assessments = []
    
    for i, feature in enumerate(should_features, 1):
        print(f"  [{i}/{len(should_features)}] Checking: {feature[:50]}...")
        result_json = await tools["assess_feature"](
            question_id,
            feature,
            "SHOULD",
            answer_text
        )
        result = json.loads(result_json)
        assessments.append(result)
        status = "✓" if result["satisfied"] else "✗"
        print(f"      {status} {result['satisfied']}")
    
    for i, feature in enumerate(shouldnt_features, 1):
        print(f"  [S{i}/{len(shouldnt_features)}] Checking: {feature[:50]}...")
        result_json = await tools["assess_feature"](
            question_id,
            feature,
            "SHOULDNT",
            answer_text
        )
        result = json.loads(result_json)
        assessments.append(result)
        status = "✓" if result["satisfied"] else "✗"
        print(f"      {status} {result['satisfied']}")
    
    print()
    
    # Step 6: Calculate score
    print("Step 6: Calculating score...")
    assessments_json = json.dumps(assessments)
    score_json = await tools["calculate_score"](assessments_json, max_score)
    score_data = json.loads(score_json)
    
    final_score = score_data["score"]
    percentage = score_data["percentage"]
    breakdown = score_data["breakdown"]
    
    print(f"✓ Score: {final_score}/{max_score} ({percentage}%)")
    print(f"✓ Breakdown: {breakdown}\n")
    
    # Step 7: Generate feedback
    print("Step 7: Generating feedback...")
    feedback_json = await tools["generate_feedback"](assessments_json)
    feedback_data = json.loads(feedback_json)
    feedback_text = feedback_data["feedback"]
    
    print("✓ Feedback generated\n")
    
    # Step 8: Save assessment
    print("Step 8: Saving assessment...")
    save_json = await tools["save_assessment"](
        question_id,
        student_code,
        final_score,
        assessments_json,
        feedback_text
    )
    save_data = json.loads(save_json)
    
    if save_data.get("status") == "success":
        print(f"✓ Saved to: {save_data['saved_to']}\n")
    else:
        print(f"✗ Error: {save_data.get('error')}\n")
    
    # Final summary
    print("="*70)
    print("ASSESSMENT SUMMARY")
    print("="*70)
    print(f"Student: {student_name} ({student_code})")
    print(f"Question: {question_text}")
    print(f"Score: {final_score}/{max_score} ({percentage}%)")
    print(f"\n{feedback_text}")
    print("="*70)


async def demo_batch_workflow():
    """Demo: Batch assessment of multiple students."""
    print("\n" + "="*70)
    print("BATCH MCP WORKFLOW DEMO")
    print("="*70)
    
    exam_dir = Path("mock_exam_submissions")
    if not exam_dir.exists():
        print(" No mock exam data found!")
        return
    
    server = ExamMCPServer(exam_dir)
    tools = server.tools
    
    print("\n MCP Server initialized\n")
    
    # Get first question
    questions_json = await tools["list_questions"]()
    questions = json.loads(questions_json)
    question = questions[0]
    question_id = question["id"]
    max_score = question["weight"]
    
    print(f"Processing question: {question_id}")
    print(f"Max score: {max_score}\n")
    
    # Get all students
    students_json = await tools["list_students"](question_id)
    students = json.loads(students_json)
    
    print(f"Found {len(students)} students\n")
    
    # Get checklist once
    checklist_json = await tools["get_checklist"](question_id)
    checklist = json.loads(checklist_json)
    should_features = checklist.get("should", [])
    shouldnt_features = checklist.get("should_not", [])
    
    results = []
    
    # Process first 3 students (to keep demo fast)
    for i, student in enumerate(students[:3], 1):
        student_code = student["code"]
        student_name = student["name"]
        
        print(f"[{i}/3] Processing: {student_name} ({student_code})")
        
        # Read answer
        answer_json = await tools["read_student_answer"](question_id, student_code)
        answer_data = json.loads(answer_json)
        answer_text = answer_data["answer"]
        
        # Assess features
        assessments = []
        for feature in should_features:
            result_json = await tools["assess_feature"](
                question_id, feature, "SHOULD", answer_text
            )
            assessments.append(json.loads(result_json))
        
        for feature in shouldnt_features:
            result_json = await tools["assess_feature"](
                question_id, feature, "SHOULDNT", answer_text
            )
            assessments.append(json.loads(result_json))
        
        # Calculate score
        score_json = await tools["calculate_score"](
            json.dumps(assessments), max_score
        )
        score_data = json.loads(score_json)
        
        results.append({
            "name": student_name,
            "code": student_code,
            "score": score_data["score"],
            "percentage": score_data["percentage"]
        })
        
        print(f"    Score: {score_data['score']}/{max_score} ({score_data['percentage']}%)\n")
    
    # Summary
    print("="*70)
    print("BATCH SUMMARY")
    print("="*70)
    for r in results:
        print(f"{r['name']:20} {r['code']:10} {r['score']:.2f}/{max_score} ({r['percentage']:.0f}%)")
    
    avg_score = sum(r["score"] for r in results) / len(results)
    avg_pct = sum(r["percentage"] for r in results) / len(results)
    print(f"\nAverage: {avg_score:.2f}/{max_score} ({avg_pct:.0f}%)")
    print("="*70)


async def demo_rag_search():
    """Demo: Search course materials."""
    print("\n" + "="*70)
    print("RAG SEARCH DEMO")
    print("="*70)
    
    server = ExamMCPServer()
    tools = server.tools
    
    if not server.vector_store:
        print(" RAG not available. Run: python fix_rag_db.py")
        return
    
    print("\n RAG vector store loaded\n")
    
    queries = [
        "what is an algorithm",
        "computer science definition",
        "difference between data and information"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        result_json = await tools["search_course_material"](query, 3)
        result = json.loads(result_json)
        
        materials = result.get("materials", [])
        print(f"Found {len(materials)} results:\n")
        
        for i, mat in enumerate(materials, 1):
            source = mat["source"]
            content = mat["content"][:100]
            print(f"  [{i}] {source}")
            print(f"      {content}...\n")
        
        print()


async def main():
    """Main menu."""
    print("\n" + "="*70)
    print("MCP TOOLS DEMO (Direct Workflow)")
    print("="*70)
    print("\nThis demo calls MCP tools directly without using an agent.")
    print("More reliable and easier to debug.\n")
    
    demos = [
        ("Simple Assessment (1 student)", demo_simple_workflow),
        ("Batch Assessment (3 students)", demo_batch_workflow),
        ("RAG Search Demo", demo_rag_search),
    ]
    
    print("Available demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    print("0. Run all demos")
    
    choice = input("\nSelect demo (0-3): ").strip()
    
    try:
        if choice == "0":
            for name, demo_func in demos:
                await demo_func()
                input("\n[Press Enter to continue]")
        elif choice in ["1", "2", "3"]:
            idx = int(choice) - 1
            await demos[idx][1]()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())