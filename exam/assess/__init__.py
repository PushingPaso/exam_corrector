from exam import QuestionsStore, Question, DIR_ROOT
from exam.llm_provider import AIOracle
from exam.solution import Answer, load_cache as load_answer
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum
from yaml import safe_dump, safe_load
from dataclasses import dataclass
import re
from exam import get_questions_store
from dataclasses import dataclass, field
import sys
import os


OUTPUT_FILE = os.getenv("OUTPUT_FILE", None)
if OUTPUT_FILE:
    OUTPUT_FILE = open(OUTPUT_FILE, "w", encoding="utf-8")
else:
    OUTPUT_FILE = sys.stdout


ALL_QUESTIONS = get_questions_store()
PATTERN_QUESTION_FOLDER = re.compile(r"^Q\d+\s+-\s+(\w+-\d+)$")
FILE_TEMPLATE = DIR_ROOT / "exam" / "assess" / "prompt-template.txt"
TEMPLATE = FILE_TEMPLATE.read_text(encoding="utf-8")

def calculate_score_from_assessments(assessments: dict, max_score: float) -> tuple[float, str, dict]:
    """
    Calcola il punteggio da un dizionario di assessment.
    Sistema: 70% Core + 30% Important
    
    Args:
        assessments: dict[Feature, FeatureAssessment]
        max_score: Punteggio massimo della domanda
        
    Returns:
        tuple(score, breakdown, stats): Punteggio, spiegazione, e statistiche dettagliate
    """
    if not assessments:
        return 0.0, "No features assessed", {}
    
    # Conta feature per tipo
    core_total = sum(1 for f in assessments if f.type == FeatureType.CORE)
    core_satisfied = sum(1 for f, a in assessments.items() 
                        if f.type == FeatureType.CORE and a.satisfied)
    
    important_total = sum(1 for f in assessments if f.type == FeatureType.DETAILS_IMPORTANT)
    important_satisfied = sum(1 for f, a in assessments.items() 
                             if f.type == FeatureType.DETAILS_IMPORTANT and a.satisfied)
    
    additional_total = sum(1 for f in assessments if f.type == FeatureType.DETAILS_ADDITIONAL)
    additional_satisfied = sum(1 for f, a in assessments.items() 
                              if f.type == FeatureType.DETAILS_ADDITIONAL and a.satisfied)
    
    # Calcolo percentuali per categoria
    core_percentage = (core_satisfied / core_total * 0.70) if core_total > 0 else 0.0
    important_percentage = (important_satisfied / important_total * 0.30) if important_total > 0 else 0.30

    # Percentuale finale
    final_percentage = core_percentage + important_percentage
    
    # Score finale
    score = round(final_percentage * max_score, 2)
    
    # Breakdown dettagliato
    breakdown_parts = []
    
    if core_total > 0:
        breakdown_parts.append(
            f"Core: {core_satisfied}/{core_total} "
            f"({core_percentage/0.70*100:.0f}% → {core_percentage*100:.0f}%)"
        )
    
    if important_total > 0:
        breakdown_parts.append(
            f"Important: {important_satisfied}/{important_total} "
            f"({important_percentage/0.20*100:.0f}% → {important_percentage*100:.0f}%)"
        )

    
    breakdown = " + ".join(breakdown_parts)
    breakdown += f" = {final_percentage*100:.0f}% of {max_score} = {score}"
    
    # Statistiche dettagliate
    stats = {
        "core": {
            "total": core_total,
            "satisfied": core_satisfied,
            "percentage": round((core_satisfied / core_total * 100) if core_total > 0 else 0, 1)
        },
        "details_important": {
            "total": important_total,
            "satisfied": important_satisfied,
            "percentage": round((important_satisfied / important_total * 100) if important_total > 0 else 0, 1)
        }
    }
    
    return score, breakdown, stats


def _load_exam(exam: Path | str | list[str] | QuestionsStore) -> QuestionsStore:
    if isinstance(exam, str):
        exam = Path(exam)
    if isinstance(exam, Path):
        questions = [d.name for d in exam.glob("Q* - *") if d.is_dir()]
        exam = []
        for question in questions:
            match = PATTERN_QUESTION_FOLDER.match(question)
            if match:
                exam.append(match.group(1))
    if isinstance(exam, list):
        if exam:
            exam = ALL_QUESTIONS.sample(*exam)
        else:
            raise ValueError("No question ID has been found or none was provided")
    if isinstance(exam, QuestionsStore):
        return exam
    else:
        raise TypeError("Exam must be a Path, str, list of question IDs, or QuestionsStore instance")


class FeatureType(str, Enum):
    """Enumeration of feature types that can be assessed in a question's answer."""
    CORE = "core"
    DETAILS_IMPORTANT = "important detail"
    DETAILS_ADDITIONAL = "additional detail"


@dataclass(frozen=True)
class Feature:
    # Type of the feature
    type: FeatureType

    # Description of the feature
    description: str

    @property
    def verb_ideal(self) -> str:
        return "should be present"

    @property
    def verb_actual(self) -> str:
        return "is actually present"
    
    @property
    def is_core(self) -> bool:
        """Determina se questa feature è core (essenziale)."""
        return self.type == FeatureType.CORE
    
    @property
    def weight_percentage(self) -> float:
        """Restituisce il peso percentuale di questa feature nel punteggio totale."""
        if self.type == FeatureType.CORE:
            return 0.70  # 70% del punteggio
        elif self.type == FeatureType.DETAILS_IMPORTANT:
            return 0.20  # 20% del punteggio
        else:  # DETAILS_ADDITIONAL
            return 0.10  # 10% del punteggio


def enumerate_features(answer: Answer):
    """Enumera le features da valutare."""
    if not answer:
        return
    i = 0
    
    # CORE - elementi essenziali
    for core_item in answer.core:
        yield i, Feature(type=FeatureType.CORE, description=core_item)
        i += 1
    
    # DETAILS_IMPORTANT - dettagli importanti
    for detail in answer.details_important:
        yield i, Feature(type=FeatureType.DETAILS_IMPORTANT, description=detail)
        i += 1



class FeatureAssessment(BaseModel):
    satisfied: bool = Field(description="Whether the feature is present in the answer")
    motivation: str = Field(description="Explanation of why the feature is present or not")


@dataclass
class AnswerAssessment:
    # The student's answer to the question.
    answer: str
    # A dictionary mapping each feature to its assessment.
    assessment: dict[Feature, FeatureAssessment] = field(default_factory=dict)

    def calculate_score(self, max_score: float) -> tuple[float, str]:
        """
        Calcola il punteggio della risposta con il nuovo sistema:
        - Core: 70% del punteggio
        - Details Important: 20% del punteggio
        - Details Additional: 10% del punteggio

        Args:
            max_score: Punteggio massimo della domanda

        Returns:
            tuple(score, breakdown): Punteggio ottenuto e spiegazione del calcolo
        """
        if not self.assessment:
            return 0.0, "No features assessed"

        # Conta feature per tipo
        core_total = sum(1 for f in self.assessment if f.type == FeatureType.CORE)
        core_satisfied = sum(1 for f, a in self.assessment.items()
                            if f.type == FeatureType.CORE and a.satisfied)

        important_total = sum(1 for f in self.assessment if f.type == FeatureType.DETAILS_IMPORTANT)
        important_satisfied = sum(1 for f, a in self.assessment.items()
                                 if f.type == FeatureType.DETAILS_IMPORTANT and a.satisfied)

        additional_total = sum(1 for f in self.assessment if f.type == FeatureType.DETAILS_ADDITIONAL)
        additional_satisfied = sum(1 for f, a in self.assessment.items()
                                  if f.type == FeatureType.DETAILS_ADDITIONAL and a.satisfied)

        # Calcolo percentuali per categoria
        core_percentage = (core_satisfied / core_total * 0.70) if core_total > 0 else 0.0
        important_percentage = (important_satisfied / important_total * 0.20) if important_total > 0 else 0.20
        additional_percentage = (additional_satisfied / additional_total * 0.10) if additional_total > 0 else 0.10

        # Percentuale finale
        final_percentage = core_percentage + important_percentage + additional_percentage

        # Score finale
        score = round(final_percentage * max_score, 2)

        # Breakdown dettagliato
        breakdown_parts = []

        if core_total > 0:
            breakdown_parts.append(f"Core: {core_satisfied}/{core_total} ({core_percentage/0.70*100:.0f}% → {core_percentage*100:.0f}%)")

        if important_total > 0:
            breakdown_parts.append(f"Important: {important_satisfied}/{important_total} ({important_percentage/0.20*100:.0f}% → {important_percentage*100:.0f}%)")

        if additional_total > 0:
            breakdown_parts.append(f"Additional: {additional_satisfied}/{additional_total} ({additional_percentage/0.10*100:.0f}% → {additional_percentage*100:.0f}%)")

        breakdown = " + ".join(breakdown_parts)
        breakdown += f" = {final_percentage*100:.0f}% of {max_score} = {score}"

        return score, breakdown


@dataclass
class StudentAssessment:
    # The name of the student.
    name: str
    # The student's code, typically an identifier or registration number.
    code: str
    # A dictionary mapping each question to its answer and assessments.
    answers: dict[Question, AnswerAssessment] = field(default_factory=dict)

    def total_score(self) -> float:
        """Calcola il punteggio totale dello studente."""
        total = 0.0
        for question, answer_assessment in self.answers.items():
            score, _ = answer_assessment.calculate_score(question.weight)
            total += score
        return round(total, 2)


class TestAssessment:
    def __init__(self):
        self.__students_by_name: dict[str, StudentAssessment] = {}
        self.__students_by_code: dict[str, StudentAssessment] = {}

    def add_assessment(self, code: str, name: str, question: Question, answer: str, feature: Feature, assessment: FeatureAssessment):
        if name not in self.__students_by_name:
            instance = StudentAssessment(
                name=name,
                code=code,
                answers={},
            )
            self.__students_by_name[name] = instance
            self.__students_by_code[code] = instance
        instance = self.__students_by_name[name]
        if question not in instance.answers:
            instance.answers[question] = AnswerAssessment(
                answer=answer,
                assessment={},
            )
        instance.answers[question].assessment[feature] = assessment

    @property
    def assessments(self) -> list[StudentAssessment]:
        return list(self.__students_by_name.values())

    def pretty_print(self, per_question: Question = None, file=OUTPUT_FILE, show_only_missing_core: bool = True):
        """
        Stampa le valutazioni in formato leggibile.
        
        Args:
            per_question: Se specificato, mostra solo questa domanda
            file: File dove scrivere l'output
            show_only_missing_core: Se True, mostra solo gli elementi CORE mancanti (default)
        """
        if per_question is not None:
            print(f"Assessments for question {per_question.id}: {per_question.text}", file=file)
        
        for index, name in enumerate(sorted(self.__students_by_name)):
            if index > 0 or per_question is not None:
                print("---", file=file)
            
            student = self.__students_by_name[name]
            print(f"Student: {student.name} ({student.code})", file=file)
            
            question_answers = student.answers
            if per_question is not None:
                question_answers = {per_question: question_answers[per_question]}
            
            for question, answer in question_answers.items():
                if per_question is None:
                    print(f"  Question: {question.text}", file=file)
                
                # Mostra preview risposta
                answer_preview = answer.answer[:200]
                if len(answer.answer) > 200:
                    answer_preview += "..."
                print(f"  Answer:\n\t{answer_preview.replace(chr(10), chr(10) + chr(9))}", file=file)
                
                # Raggruppa features per tipo
                core_missing = []
                important_missing = []
                important_present = []

                
                for feature, assessment in answer.assessment.items():
                    if feature.type == FeatureType.CORE:
                        if not assessment.satisfied:
                            core_missing.append((feature, assessment))
                    elif feature.type == FeatureType.DETAILS_IMPORTANT:
                        if not assessment.satisfied:
                            important_missing.append((feature, assessment))
                        else:
                            important_present.append((feature, assessment))

                
                # Mostra problemi CORE
                if core_missing:
                    print("Missing CORE elements:", file=file)
                    for feature, assessment in core_missing:
                        print(f"{feature.description}", file=file)
                        print(f"{assessment.motivation.replace(chr(10), chr(10) + '        → ')}", file=file)
                else:
                    print("All CORE elements satisfied!", file=file)
                
                # Mostra dettagli solo se richiesto
                if not show_only_missing_core:
                    if important_missing or important_present:
                        print("  Important Details:", file=file)
                        for feature, assessment in important_present:
                            print(f"{feature.description}", file=file)
                        for feature, assessment in important_missing:
                            print(f"{feature.description}", file=file)
                            print(f"{assessment.motivation.replace(chr(10), chr(10) + '        → ')}", file=file)

                
                # Calcola e mostra punteggio
                score, breakdown = answer.calculate_score(3)
                print(f"  Score: {score}/{question.weight}", file=file)
                print(f"  Calculation: {breakdown}", file=file)
            
            # Mostra punteggio totale se più domande
            if len(student.answers) > 1:
                total = student.total_score()
                max_total = sum(q.weight for q in student.answers.keys())
                print(f"\n  Total Score: {total}/{max_total}", file=file)


def first(iterable):
    """Return the first item of an iterable or None if it's empty."""
    return next(iter(iterable), None)


class Assessor(AIOracle):
    def __init__(self, exam_dir_by_questions: Path, model_name: str = None, model_provider: str = None):
        """
        Inizializza l'assessor.
        
        Args:
            exam_dir_by_questions: Directory contenente le risposte degli studenti
            model_name: Nome del modello LLM da usare
            model_provider: Provider del modello
        """
        super().__init__(model_name, model_provider, FeatureAssessment)
        self.__root = Path(exam_dir_by_questions)
        self.__exam: QuestionsStore = _load_exam(exam_dir_by_questions)
        self.__answers: dict[str, Answer] = {}
        
        for question in self.__exam.questions:
            if (cached_answer := load_answer(question)):
                self.__answers[question.id] = cached_answer
            else:
                raise ValueError(f"Cached answer for question {question.id} not found")

    def __iterate_over_answers(self, on_question_over: callable = None):
        for question in self.__exam.questions:
            target = self.__answers.get(question.id)
            for dir in self.__root.glob(f"Q* - {question.id}/*"):
                if dir.is_dir():
                    code, name = dir.name.split(" - ")[:2]
                    attempt_file = first(dir.glob("Attempt*_textresponse"))
                    answer = attempt_file.read_text(encoding="utf-8") if attempt_file else None
                    yield (code, name, question, target, answer, dir)
            if on_question_over:
                on_question_over(question)

    def __save_cache(self, dir: Path, feature: Feature, assessment: FeatureAssessment, index: int):
        if not dir:
            return
        cache_file = dir / f"feature_{index}_{feature.type.name}.yml"
        cache_data = assessment.model_dump()
        cache_data["feature"] = feature.description
        cache_data["feature_type"] = feature.type.name
        with cache_file.open("w", encoding="utf-8") as f:
            safe_dump(cache_data, f, sort_keys=True, allow_unicode=True)
        print(f"# saved assessment to {cache_file}")

    def __load_cache(self, dir: Path, feature: Feature, index: int) -> FeatureAssessment | None:
        if not dir:
            return None
        cache_file = dir / f"feature_{index}_{feature.type.name}.yml"
        if not cache_file.exists():
            return None
        with cache_file.open("r", encoding="utf-8") as f:
            try:
                cached_data = safe_load(f)
                return FeatureAssessment(
                    motivation=cached_data.get("motivation"),
                    satisfied=cached_data.get("satisfied", False),
                )
            except Exception as e:
                print(f"# error loading cached assessment from {cache_file}: {e}")
                cache_file.unlink()

    def __assess_feature(self, question: Question, feature: Feature, answer: str, dir: Path, index: int) -> FeatureAssessment:
        cached_assessment = self.__load_cache(dir, feature, index)
        if cached_assessment:
            print(f"loaded cached assessment for {feature.type.name} from {dir}")
            return cached_assessment
        prompt = TEMPLATE.format(
            class_name=FeatureAssessment.__name__,
            question=question.text,
            feature_type=feature.type.value,
            feature_verb_ideal=feature.verb_ideal,
            feature_verb_actual=feature.verb_actual,
            feature=feature.description,
            answer=answer
        )
        result = self.llm.invoke(prompt)
        if not isinstance(result, FeatureAssessment):
            raise TypeError(f"Expected {FeatureAssessment.__name__}, got {type(result)}")
        self.__save_cache(dir, feature, result, index)
        return result

    def assess_all(self, show_only_missing_core: bool = True):
        """
        Valuta tutte le risposte.
        
        Args:
            show_only_missing_core: Se True, mostra solo elementi CORE mancanti (default).
                                   Se False, mostra tutte le valutazioni.
        """
        assessments = TestAssessment()
        
        def log(q):
            assessments.pretty_print(per_question=q, file=OUTPUT_FILE, show_only_missing_core=show_only_missing_core)
        
        total_features = 0
        evaluated_features = 0
        
        for code, name, question, target, answer, dir in self.__iterate_over_answers(on_question_over=log):
            for index, feature in enumerate_features(target):
                total_features += 1
                assessment = self.__assess_feature(question, feature, answer, dir, index)
                assessments.add_assessment(code, name, question, answer, feature, assessment)
                evaluated_features += 1
        
        print(f"\n# Total features evaluated: {evaluated_features}", file=OUTPUT_FILE)
        
        return assessments