from exam import QuestionsStore, Question, DIR_ROOT
from exam.openai import AIOracle
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
    EXAMPLE = "example"
    SHOULD = "mandatory mention"
    SHOULDNT = "mistake"
    SEE_ALSO = "optional mention"


@dataclass(frozen=True)
class Feature:
    # Type of the feature, indicating whether it is an example, a mandatory mention, a mistake, or an optional mention.
    type: FeatureType

    # Description of the feature, typically a string that explains what the feature is about.
    description: str

    @property
    def verb_ideal(self) -> str:
        if self.type != FeatureType.SHOULDNT:
            return "should be present"
        return "should be absent"

    @property
    def verb_actual(self) -> str:
        if self.type != FeatureType.SHOULDNT:
            return "is actually present"
        return "is actually absent"
    
    @property
    def is_critical(self) -> bool:
        """Determina se questa feature è critica e deve sempre essere mostrata."""
        return self.type in (FeatureType.SHOULD, FeatureType.SHOULDNT)


def enumerate_features(answer: Answer, only_critical: bool = False):
    """
    Enumera le features da valutare.
    
    Args:
        answer: Answer object con le features
        only_critical: Se True, enumera solo SHOULD e SHOULDNT (ignora EXAMPLE e SEE_ALSO)
    """
    if not answer:
        return
    i = 0
    
    # SHOULD - sempre incluse
    for should in answer.should:
        yield i, Feature(type=FeatureType.SHOULD, description=should)
        i += 1
    
    # SHOULDNT - sempre incluse
    for shouldnt in answer.should_not:
        yield i, Feature(type=FeatureType.SHOULDNT, description=shouldnt)
        i += 1
    
    # EXAMPLE e SEE_ALSO - solo se richiesto

    for example in answer.examples:
        yield i, Feature(type=FeatureType.EXAMPLE, description=example)
        i += 1

    for see_also in answer.see_also:
        yield i, Feature(type=FeatureType.SEE_ALSO, description=see_also)
        i += 1


class FeatureAssessment(BaseModel):
    satisfied: bool = Field(description="Whether the feature is present (if it should be present) or lacking (if it shouldn't be present)")
    motivation: str = Field(description="Explanation of why the feature is present or not")


@dataclass
class AnswerAssessment:
    # The student's answer to the question.
    answer: str
    # A dictionary mapping each feature to its assessment.
    assessment: dict[Feature, FeatureAssessment] = field(default_factory=dict)
    
    def calculate_score(self, max_score: float) -> tuple[float, str]:
        """
        Calcola il punteggio della risposta.
        
        Args:
            max_score: Punteggio massimo della domanda
            
        Returns:
            tuple(score, breakdown): Punteggio ottenuto e spiegazione del calcolo
        """
        if not self.assessment:
            return 0.0, "No features assessed"
        
        # Conta feature per tipo
        should_total = sum(1 for f in self.assessment if f.type == FeatureType.SHOULD)
        should_satisfied = sum(1 for f, a in self.assessment.items() 
                              if f.type == FeatureType.SHOULD and a.satisfied)
        
        shouldnt_total = sum(1 for f in self.assessment if f.type == FeatureType.SHOULDNT)
        shouldnt_violated = sum(1 for f, a in self.assessment.items() 
                               if f.type == FeatureType.SHOULDNT and not a.satisfied)
        
        if should_total == 0:
            return 0.0, "No SHOULD features defined"
        
        # Calcolo base: percentuale di SHOULD soddisfatti
        base_percentage = should_satisfied / should_total
        
        # Penalità per errori (SHOULDNT violati)
        penalty_per_error = 0.15  # 15% di penalità per errore
        error_penalty = shouldnt_violated * penalty_per_error
        
        # Percentuale finale
        final_percentage = max(0.0, base_percentage - error_penalty)
        
        # Score finale
        score = round(final_percentage * max_score, 2)
        
        # Breakdown
        breakdown = f"Base: {should_satisfied}/{should_total} SHOULD ({base_percentage*100:.0f}%)"
        if shouldnt_violated > 0:
            breakdown += f" - Errors: {shouldnt_violated}/{shouldnt_total} mistakes (-{error_penalty*100:.0f}%)"
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

    def pretty_print(self, per_question: Question = None, file=OUTPUT_FILE, show_only_unsatisfied: bool = True):
        """
        Stampa le valutazioni in formato leggibile.
        
        Args:
            per_question: Se specificato, mostra solo questa domanda
            file: File dove scrivere l'output
            show_only_unsatisfied: Se True, mostra solo le feature non soddisfatte (default)
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
                
                # Filtra le features da mostrare
                features_to_show = []
                for feature, assessment in answer.assessment.items():
                    # Mostra solo SHOULD e SHOULDNT
                    if feature.is_critical:
                        # Se show_only_unsatisfied, mostra solo quelle non soddisfatte
                        if show_only_unsatisfied:
                            # Per SHOULD: mostra se NOT satisfied (mancante)
                            # Per SHOULDNT: mostra se NOT satisfied (errore presente)
                            if not assessment.satisfied:
                                features_to_show.append((feature, assessment))
                        else:
                            # Mostra tutto
                            features_to_show.append((feature, assessment))
                
                if features_to_show:
                    print("  Issues Found:", file=file)
                    for feature, assessment in features_to_show:
                        if feature.type == FeatureType.SHOULD:
                            print(f"    ✗ Missing: {feature.description}", file=file)
                        else:  # SHOULDNT
                            print(f"    ✗ Error: {feature.description}", file=file)
                        print(f"        → {assessment.motivation.replace(chr(10), chr(10) + '        → ')}", file=file)
                else:
                    if show_only_unsatisfied:
                        print("  ✓ All critical requirements satisfied!", file=file)
                    else:
                        print("  ✓ No issues found", file=file)
                
                # Calcola e mostra punteggio
                score, breakdown = answer.calculate_score(question.weight)
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
    def __init__(self, exam_dir_by_questions: Path, model_name: str = None, model_provider: str = None, 
                 only_critical_features: bool = True):
        """
        Inizializza l'assessor.
        
        Args:
            exam_dir_by_questions: Directory contenente le risposte degli studenti
            model_name: Nome del modello LLM da usare
            model_provider: Provider del modello
            only_critical_features: Se True, valuta solo SHOULD e SHOULDNT
        """
        super().__init__(model_name, model_provider, FeatureAssessment)
        self.__root = Path(exam_dir_by_questions)
        self.__exam: QuestionsStore = _load_exam(exam_dir_by_questions)
        self.__answers: dict[str, Answer] = {}
        self.__only_critical = only_critical_features
        
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
            print(f"# loaded cached assessment for {feature.type.name} from {dir}")
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

    def assess_all(self, show_only_unsatisfied: bool = True):
        """
        Valuta tutte le risposte.
        
        Args:
            show_only_unsatisfied: Se True, mostra solo problemi (default).
                                  Se False, mostra tutte le valutazioni.
        """
        assessments = TestAssessment()
        
        def log(q):
            assessments.pretty_print(per_question=q, file=OUTPUT_FILE, show_only_unsatisfied=show_only_unsatisfied)
        
        total_features = 0
        evaluated_features = 0
        
        for code, name, question, target, answer, dir in self.__iterate_over_answers(on_question_over=log):
            for index, feature in enumerate_features(target, only_critical=self.__only_critical):
                total_features += 1
                assessment = self.__assess_feature(question, feature, answer, dir, index)
                assessments.add_assessment(code, name, question, answer, feature, assessment)
                evaluated_features += 1
        
        print(f"\n# Total features evaluated: {evaluated_features}", file=OUTPUT_FILE)
        
        return assessments