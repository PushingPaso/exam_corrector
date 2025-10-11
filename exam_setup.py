"""
Script helper per preparare gli esami per la correzione automatica.
Uso: python exam_setup.py
"""

import yaml
from pathlib import Path
from exam import DIR_ROOT, get_questions_store

EXAMS_DIR = DIR_ROOT / "static" / "se-exams"


def check_exam_files():
    """Verifica quali file di esame sono disponibili."""
    print("="*70)
    print("EXAM FILES CHECK")
    print("="*70)
    
    EXAMS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDirectory: {EXAMS_DIR}")
    
    yaml_files = list(EXAMS_DIR.glob("*.yml")) + list(EXAMS_DIR.glob("*.yaml"))
    
    if not yaml_files:
        print("\nNo YAML files found!")
        print(f"Please place your exam YAML files in: {EXAMS_DIR}")
        return []
    
    print(f"\nFound {len(yaml_files)} YAML files:")
    for f in yaml_files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    # Group by exam
    exam_pairs = {}
    for f in yaml_files:
        name = f.name
        if "questions" in name:
            base = name.replace("-questions", "").replace(".yml", "").replace(".yaml", "")
            if base not in exam_pairs:
                exam_pairs[base] = {}
            exam_pairs[base]["questions"] = f
        elif "responses" in name:
            base = name.replace("-responses", "").replace(".yml", "").replace(".yaml", "")
            if base not in exam_pairs:
                exam_pairs[base] = {}
            exam_pairs[base]["responses"] = f
    
    print(f"\nExam pairs found: {len(exam_pairs)}")
    for base, files in exam_pairs.items():
        has_questions = "questions" in files
        has_responses = "responses" in files
        status = "Complete" if has_questions and has_responses else "Incomplete"
        print(f"  - {base}: {status}")
        if not has_questions:
            print(f"    Missing: {base}-questions.yml")
        if not has_responses:
            print(f"    Missing: {base}-responses.yml")
    
    return [base for base, files in exam_pairs.items() 
            if "questions" in files and "responses" in files]


def extract_question_ids(exam_base: str):
    """Estrae gli ID delle domande da un file questions."""
    questions_file = EXAMS_DIR / f"{exam_base}-questions.yml"
    
    if not questions_file.exists():
        print(f"Error: {questions_file} not found")
        return []
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    question_ids = []
    for key, value in data.items():
        if key.startswith("Question"):
            q_id = value.get("id")
            if q_id:
                question_ids.append(q_id)
    
    return question_ids


def check_checklists(question_ids):
    """Verifica quali checklist sono disponibili."""
    print("\n" + "="*70)
    print("CHECKLISTS CHECK")
    print("="*70)
    
    solutions_dir = DIR_ROOT / "solutions"
    
    if not solutions_dir.exists():
        print(f"\nSolutions directory not found: {solutions_dir}")
        return []
    
    missing = []
    found = []
    
    for q_id in question_ids:
        checklist_file = solutions_dir / f"{q_id}.yaml"
        if checklist_file.exists():
            found.append(q_id)
            print(f"  OK: {q_id}")
        else:
            missing.append(q_id)
            print(f"  MISSING: {q_id}")
    
    print(f"\nSummary: {len(found)}/{len(question_ids)} checklists found")
    
    return missing


def generate_missing_checklists(question_ids):
    """Genera le checklist mancanti."""
    if not question_ids:
        print("\nAll checklists are present!")
        return
    
    print(f"\nGenerate {len(question_ids)} missing checklists? (y/n): ", end="")
    response = input().strip().lower()
    
    if response != 'y':
        print("Skipped.")
        return
    
    from exam.solution import SolutionProvider
    
    print("\nGenerating checklists...")
    llm = SolutionProvider()
    questions_store = get_questions_store()
    
    for q_id in question_ids:
        try:
            print(f"  Generating {q_id}...", end=" ")
            question = questions_store.question(q_id)
            answer = llm.answer(question)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")


def main():
    print("\nEXAM SETUP HELPER")
    print("="*70)
    
    # Check exam files
    complete_exams = check_exam_files()
    
    if not complete_exams:
        print("\nNo complete exam pairs found.")
        print(f"Please add both questions and responses files to: {EXAMS_DIR}")
        return
    
    # Select exam
    if len(complete_exams) == 1:
        exam_base = complete_exams[0]
        print(f"\nUsing exam: {exam_base}")
    else:
        print("\nSelect exam to setup:")
        for i, base in enumerate(complete_exams, 1):
            print(f"  {i}. {base}")
        choice = input("Choice (1-{}): ".format(len(complete_exams))).strip()
        try:
            exam_base = complete_exams[int(choice) - 1]
        except:
            print("Invalid choice")
            return
    
    # Extract question IDs
    print(f"\nExtracting question IDs from {exam_base}-questions.yml...")
    question_ids = extract_question_ids(exam_base)
    
    if not question_ids:
        print("No questions found in file!")
        return
    
    print(f"Found {len(question_ids)} questions:")
    for q_id in question_ids:
        print(f"  - {q_id}")
    
    # Check checklists
    missing_checklists = check_checklists(question_ids)
    
    # Generate if needed
    if missing_checklists:
        generate_missing_checklists(missing_checklists)
    
    # Final summary
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    print(f"\nExam ready for correction:")
    print(f"  Questions: {exam_base}-questions.yml")
    print(f"  Responses: {exam_base}-responses.yml")
    print(f"  Location: {EXAMS_DIR}")
    print(f"\nTo correct the exam, run:")
    print(f'  python mcp_client.py')
    print(f'  Then: "Correct the {exam_base} exam"')


if __name__ == "__main__":
    main()