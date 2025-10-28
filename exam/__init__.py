import csv
import xml
from dataclasses import dataclass
import xml.etree.ElementTree as xml
from pathlib import Path
from io import StringIO
from markdown import markdown

DIR_ROOT = Path(__file__).parent.parent
DEFAULT_QUESTIONS_FILE = DIR_ROOT / "static" / "questions.csv"

_QUESTIONS_STORE_INSTANCE = None


def get_questions_store(questions=DEFAULT_QUESTIONS_FILE):
    """
    Get or create the singleton QuestionsStore instance.
    This ensures all parts of the system use the same instance.
    """
    global _QUESTIONS_STORE_INSTANCE
    if _QUESTIONS_STORE_INSTANCE is None:
        _QUESTIONS_STORE_INSTANCE = QuestionsStore(questions)
    return _QUESTIONS_STORE_INSTANCE


def load_exam_from_yaml(questions_file: str, responses_file: str, grades_file: str = None, exams_dir=None):
    """
    Load an entire exam from YAML files.

    Args:
        questions_file: Filename or path to questions YAML
        responses_file: Filename or path to responses YAML
        grades_file: Optional filename or path to grades YAML
        exams_dir: Directory containing YAML files (default: static/se-exams)

    Returns:
        dict with:
            - exam_id: str
            - questions: list of question dicts
            - students: list of student dicts with responses
            - files: dict with file paths used

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If YAML parsing fails
    """
    import yaml
    import re
    from pathlib import Path

    if exams_dir is None:
        exams_dir = DIR_ROOT / "static" / "se-exams"
    else:
        exams_dir = Path(exams_dir)

    # Resolve file paths
    questions_path = Path(questions_file)
    if not questions_path.is_absolute():
        questions_path = exams_dir / questions_file

    responses_path = Path(responses_file)
    if not responses_path.is_absolute():
        responses_path = exams_dir / responses_file

    # Check existence
    if not questions_path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_path}\n"
            f"Searched in: {exams_dir}"
        )

    if not responses_path.exists():
        raise FileNotFoundError(
            f"Responses file not found: {responses_path}\n"
            f"Searched in: {exams_dir}"
        )

    # Load YAML files
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions_data = yaml.safe_load(f)

    with open(responses_path, 'r', encoding='utf-8') as f:
        responses_data = yaml.safe_load(f)

    # Load grades file if provided
    grades_data = None
    grades_path = None
    if grades_file:
        grades_path = Path(grades_file)
        if not grades_path.is_absolute():
            grades_path = exams_dir / grades_file

        if grades_path.exists():
            with open(grades_path, 'r', encoding='utf-8') as f:
                grades_data = yaml.safe_load(f)
        else:
            print(f"[LOAD_EXAM] Warning: grades file not found: {grades_path}")

    # Parse grades into email-indexed dict
    grades_by_email = {}
    if grades_data:
        question_grade_pattern = re.compile(r'^q(\d+)\d{3}$')

        for grade_entry in grades_data:
            email = grade_entry.get("emailaddress")
            if email and grade_entry.get("state") == "Finished":
                question_grades = {}

                for key, value in grade_entry.items():
                    match = question_grade_pattern.match(key)
                    if match:
                        question_num = int(match.group(1))
                        try:
                            question_grades[question_num] = float(value)
                        except (ValueError, TypeError):
                            pass

                grades_by_email[email] = {
                    "total_grade": float(grade_entry.get("grade2700", 0)),
                    "question_grades": question_grades
                }

    # Parse questions
    questions = []
    for key, value in questions_data.items():
        if key.startswith("Question"):
            questions.append({
                "number": key,
                "id": value.get("id"),
                "text": value.get("text"),
                "score": value.get("score", 3.0)
            })

    # Parse students
    students = []
    for student_data in responses_data:
        if student_data.get("state") != "Finished":
            continue

        email = student_data.get("emailaddress", "unknown")

        # Extract responses
        responses = {}
        for i in range(1, len(questions) + 1):
            response_key = f"response{i}"
            if response_key in student_data:
                response_text = student_data[response_key]
                if response_text and response_text.strip() != '-':
                    responses[i] = response_text

        students.append({
            "email": email,
            "started": student_data.get("startedon"),
            "completed": student_data.get("completed"),
            "time_taken": student_data.get("timetaken"),
            "moodle_grade": student_data.get("grade2700"),
            "responses": responses,
            "num_responses": len(responses),
            "original_grades": grades_by_email.get(email, {})
        })

    # Generate exam ID
    exam_id = f"{questions_path.stem}_{responses_path.stem}"

    return {
        "exam_id": exam_id,
        "questions": questions,
        "students": students,
        "files": {
            "questions": str(questions_path),
            "responses": str(responses_path),
            "grades": str(grades_path) if grades_path else None
        }
    }


class IdGenerator:
    def __init__(self):
        self.__categories = dict()

    def id_for(self, category):
        if category not in self.__categories:
            self.__categories[category] = 1
        else:
            self.__categories[category] += 1
        return f"{category}-{self.__categories[category]}"


DEFAULT_ID_GENERATOR = IdGenerator()


@dataclass(unsafe_hash=True)
class Category:
    name: str

    def __post_init__(self):
        self.name = self.name.strip().replace(" ", "")

    def copy(self):
        return Category(self.name)

    def to_xml(self, root: xml.Element):
        if root is None:
            root = xml.Element("question")
        else:
            root = xml.SubElement(root, "question")
        root.set("type", "category")
        category = xml.SubElement(root, "category")
        xml.SubElement(category, "text").text = f'$course$/top/{self.name}'
        info = xml.SubElement(root, "info")
        info.set("format", "html")
        xml.SubElement(info, "text").text = ""
        return root


@dataclass(unsafe_hash=True)
class Question:
    category: Category = Category("Default")
    text: str = ""
    type: str = "essay"
    weight: float = 1.0
    max_lines: int = 15
    id: str = None

    def __post_init__(self):
        if not isinstance(self.category, Category):
            self.category = Category(self.category)
        if self.id is None:
            self.id = DEFAULT_ID_GENERATOR.id_for(self.category.name)
        self.weight = float(self.weight)
        self.max_lines = int(self.max_lines)

    def copy(self):
        return Question(
            category=self.category.copy(),
            text=self.text,
            type=self.type,
            weight=self.weight,
            max_lines=self.max_lines,
            id=self.id,
        )

    def to_xml(self, root: xml.Element):
        if root is None:
            root = xml.Element("question")
        else:
            root = xml.SubElement(root, "question")
        root.set("type", self.type)
        name = xml.SubElement(root, "name")
        xml.SubElement(name, "text").text = self.id
        questiontext = xml.SubElement(root, "questiontext")
        questiontext.set("format", "html")
        xml.SubElement(questiontext, "text").text = markdown(self.text)
        xml.SubElement(root, "defaultgrade").text = str(float(self.weight))
        xml.SubElement(root, "penalty").text = "0"
        xml.SubElement(root, "hidden").text = "0"
        xml.SubElement(root, "responserequired").text = "1"
        xml.SubElement(root, "responseformat").text = "editor"
        xml.SubElement(root, "responsefieldlines").text = str(int(self.max_lines))
        xml.SubElement(root, "attachments").text = "0"
        xml.SubElement(root, "attachmentsrequired").text = "0"
        return root


def load_questions_from_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            yield Question(
                category=row["Category"],
                text=row["Question"],
                weight=row["Weight"],
            )


def group_by_category(questions):
    questions_by_category = dict()
    for question in questions:
        questions_by_category.setdefault(question.category, []).append(question)
    return questions_by_category


class QuestionsStore:
    def __init__(self, questions=DEFAULT_QUESTIONS_FILE):
        if isinstance(questions, Path) or isinstance(questions, str):
            questions = load_questions_from_csv(questions)
        questions = [q.copy() for q in questions]
        self.__questions_by_category = group_by_category(questions)
        self.__questions_by_id = {q.id: q for question_list in self.__questions_by_category.values() for q in
                                  question_list}
        self.__categories = tuple(sorted(self.__questions_by_category.keys(), key=lambda x: x.name))

    @property
    def categories(self):
        return sorted(self.__categories, key=lambda x: x.name)

    @property
    def questions(self):
        return sorted(self.__questions_by_id.values(), key=lambda x: x.id)

    def category(self, category):
        if not isinstance(category, Category):
            category = Category(category)
        if category not in self.__categories:
            raise KeyError(f"Category {category} not found")
        return category

    def question(self, id):
        # Prova ricerca esatta
        if id in self.__questions_by_id:
            return self.__questions_by_id[id]

        # Prova ricerca case-insensitive
        id_lower = id.lower()
        for qid, question in self.__questions_by_id.items():
            if qid.lower() == id_lower:
                return question

        # Prova ricerca fuzzy (rimuovi spazi, trattini)
        id_normalized = id.replace(" ", "").replace("-", "").lower()
        for qid, question in self.__questions_by_id.items():
            qid_normalized = qid.replace(" ", "").replace("-", "").lower()
            if qid_normalized == id_normalized:
                return question

        # Se ancora non trovato, mostra quali sono disponibili
        available_ids = list(self.__questions_by_id.keys())
        raise KeyError(
            f"Question '{id}' not found. Available IDs: {available_ids[:10]}"
            + (f"... and {len(available_ids) - 10} more" if len(available_ids) > 10 else "")
        )

    def questions_in_category(self, category):
        category = self.category(category)
        return sorted(self.__questions_by_category.get(category, []), key=lambda x: x.id)

    def category_size(self, category):
        category = self.category(category)
        return len(self.__questions_by_category.get(category, []))

    def category_weight(self, category):
        category = self.category(category)
        return sum(q.weight for q in self.__questions_by_category.get(category, []))

    def __len__(self):
        return len(self.questions)

    def __total_weight(self):
        return sum(q.weight for q in self.__questions_by_id.values())

    def sample(self, id: str, *others: str) -> 'QuestionsStore':
        ids = [id] + list(others)
        questions = [self.question(q_id) for q_id in ids]
        return QuestionsStore(questions)

    @property
    def total_weight(self):
        return self.__total_weight()

    @total_weight.setter
    def total_weight(self, value):
        old_weight = self.__total_weight()
        if value == old_weight:
            return
        factor = value / old_weight
        for question in self.questions:
            question.weight *= factor

    def to_xml(self, rootname="quiz"):
        quiz = xml.Element(rootname)
        for category in self.categories:
            category.to_xml(quiz)
            for question in self.questions_in_category(category):
                question.to_xml(quiz)
        return xml.ElementTree(quiz)

    def __str__(self):
        result = StringIO()
        print(f"# {len(self)} questions, total weight: {self.total_weight:.2f}", file=result)
        for category in self.categories:
            print(
                f"## {category.name} ({self.category_size(category)} questions, total weight: {self.category_weight(category):.2f})",
                file=result)
            for question in self.questions_in_category(category):
                print(f"- {question.id} ({question.weight:.2f}): {question.text}", file=result)
        return result.getvalue()

    def __repr__(self):
        return f"QuestionsStore({self.questions})"

    def __eq__(self, value):
        if not isinstance(value, QuestionsStore):
            return False
        return self.questions == value.questions

    def __hash__(self):
        return hash(self.questions)