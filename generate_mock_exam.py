"""
Script per generare esami di test fittizi per testare il sistema di correzione.
Crea una struttura di directory con risposte simulate di studenti.
"""

from pathlib import Path
import random
from exam import QuestionsStore

# Risposte simulate per diversi livelli di qualità
MOCK_ANSWERS = {
    "excellent": [
        """Build automation is the process of automatically executing tasks involved in creating software builds, including compiling source code, running tests, packaging artifacts, and preparing deployments without manual intervention.

**Key aspects of build automation:**
- **Consistency**: Ensures reproducible builds across different environments
- **Efficiency**: Eliminates manual errors and reduces build time
- **Integration**: Seamlessly works with version control and CI/CD pipelines
- **Scalability**: Can handle complex projects with multiple dependencies

Build automation tools typically provide dependency management, incremental compilation, automated testing integration, and deployment capabilities. They form the backbone of modern DevOps practices by enabling continuous integration and delivery.""",

        """The build life cycle of a software project represents the sequence of phases that transform source code into deployable artifacts:

**1. Initialization**: Environment setup, configuration validation, and workspace preparation
**2. Pre-processing**: Code generation, resource processing, and dependency validation
**3. Compilation**: Source code compilation, bytecode generation, and syntax checking
**4. Post-processing**: Code optimization, bytecode enhancement, and resource bundling
**5. Testing**: Unit test execution, integration testing, and code coverage analysis
**6. Packaging**: Artifact creation (JARs, WARs, executables), documentation generation
**7. Verification**: Integration tests, quality gates, and security scans
**8. Installation/Deployment**: Artifact deployment to repositories or target environments

Each phase has specific goals, produces intermediate outputs, and may trigger subsequent phases based on success criteria. This structured approach ensures quality and traceability throughout the build process.""",

        """From a build automation tool, you can expect comprehensive functionality that streamlines the entire software delivery process:

**Core Capabilities:**
- **Dependency Management**: Automatic resolution, downloading, and version management of libraries
- **Multi-platform Support**: Cross-platform builds for different operating systems and architectures
- **Incremental Building**: Smart rebuilds of only changed components for efficiency
- **Parallel Execution**: Concurrent task execution to optimize build time

**Quality Assurance:**
- **Automated Testing**: Integration with testing frameworks for unit, integration, and acceptance tests
- **Code Quality Checks**: Static analysis, code coverage metrics, and coding standards enforcement
- **Security Scanning**: Vulnerability detection in dependencies and code

**DevOps Integration:**
- **CI/CD Pipeline Support**: Seamless integration with continuous integration systems
- **Artifact Management**: Publishing to repositories and artifact stores
- **Environment Management**: Configuration for different deployment targets
- **Reporting**: Comprehensive build logs, test results, and quality metrics"""
    ],
    
    "good": [
        """Build automation refers to the process of automatically compiling, testing, and packaging software applications without manual intervention.

In software engineering, build automation eliminates repetitive manual tasks by automatically executing the steps needed to create a software build. This includes compiling source code, resolving dependencies, running tests, and creating deployable artifacts.

Build automation is important because it:
- Reduces human errors in the build process
- Saves time for developers
- Ensures consistent builds across different environments
- Enables continuous integration practices

Common build automation tools include Maven and Gradle for Java, npm for JavaScript, and Make for C/C++ projects.""",

        """The build life cycle of a software project consists of several sequential phases that transform source code into a deployable application:

**Main phases:**
1. **Compilation**: Convert source code into executable format
2. **Testing**: Run automated tests to verify functionality
3. **Packaging**: Create deployable artifacts (JAR files, executables)
4. **Deployment**: Install the application to target environments

Each phase depends on the successful completion of previous phases. For example, testing can only occur after successful compilation, and packaging happens after tests pass.

The build life cycle ensures that software is built in a consistent, repeatable manner and helps catch issues early in the development process.""",

        """From a build automation tool, you can generally expect several key capabilities:

**Core Features:**
- Automatic compilation of source code
- Dependency management and resolution
- Integration with testing frameworks
- Creation of deployable artifacts

**Additional Benefits:**
- Build scripts and configuration management
- Support for different environments (development, testing, production)
- Integration with version control systems
- Build reporting and logging
- Error detection and reporting

These tools streamline the development workflow by automating repetitive tasks and ensuring that builds are consistent and reliable. They are essential for modern software development practices."""
    ],
    
    "sufficient": [
        """Build automation is the process of automatically compiling and building software projects without doing it manually.

In software engineering, build automation helps developers by automatically performing tasks like compiling code, running tests, and creating the final application. This saves time and reduces errors that can happen when doing these steps by hand.""",

        """The build life cycle is the sequence of steps that happen when building a software project.

The main steps include:
1. Compiling the source code
2. Running tests to check if everything works
3. Packaging the application into a usable format
4. Deploying it to where it needs to run

These steps happen in order, and each step must succeed before moving to the next one.""",

        """From a build automation tool, you can expect it to help with several important tasks:

- Automatically compile your source code
- Run tests to make sure the code works correctly
- Manage dependencies and libraries your project needs
- Create the final application files
- Help deploy the application

These tools make development easier and faster by doing repetitive tasks automatically."""
    ],
    
    "insufficient": [
        """Build automation is about automatically building software. It's used in software engineering to make things easier for developers, but I'm not sure exactly how it works.""",

        """The build life cycle has something to do with the steps to build software. There are different phases like compilation and testing, but I don't remember the exact order or what each one does.""",

        """Build automation tools help with building software. They can do things automatically instead of manually, but I'm not sure what specific features they have or what you can expect from them."""
    ],
    
    "wrong": [
        """Build automation is a project management methodology like Agile or Scrum. In software engineering, it's about organizing teams and planning sprints to build software faster. It focuses on communication and collaboration between developers.""",

        """The build life cycle is the same as the software development life cycle. It includes phases like requirements gathering, design, implementation, testing, and maintenance. It's about managing the entire project from start to finish over months or years.""",

        """Build automation tools are basically IDEs like Visual Studio or Eclipse. You can expect them to have features like syntax highlighting, debugging, code completion, and project management. They help you write code faster and find bugs in your programs."""
    ]
}

# Nomi fittizi di studenti
STUDENT_NAMES = [
    "Mario Rossi", "Luigi Bianchi", "Giovanni Verdi", "Anna Ferrari",
    "Marco Colombo", "Laura Russo", "Giuseppe Romano", "Francesca Marino",
    "Alessandro Greco", "Sofia Conti", "Matteo Ricci", "Giulia Bruno",
    "Davide Costa", "Chiara Gallo", "Andrea Fontana", "Elena Lombardi",
    "Simone Villa", "Valentina Moretti", "Lorenzo Barbieri", "Sara Rizzo"
]

def generate_student_code():
    """Genera un codice studente fittizio."""
    return str(random.randint(100000, 999999))

def get_random_answer(quality_distribution=None):
    """
    Seleziona una risposta casuale in base alla distribuzione di qualità.
    
    Args:
        quality_distribution: dict con percentuali per ogni livello
                             Default: distribuita realisticamente
    """
    if quality_distribution is None:
        quality_distribution = {
            "excellent": 15,
            "good": 35,
            "sufficient": 30,
            "insufficient": 15,
            "wrong": 5
        }
    
    # Normalizza le percentuali
    total = sum(quality_distribution.values())
    normalized = {k: v/total for k, v in quality_distribution.items()}
    
    # Seleziona livello in base alle probabilità
    rand = random.random()
    cumulative = 0
    for level, prob in normalized.items():
        cumulative += prob
        if rand <= cumulative:
            return random.choice(MOCK_ANSWERS[level])
    
    return random.choice(MOCK_ANSWERS["sufficient"])

def create_mock_exam_structure(
    output_dir: str = "mock_exam_submissions",
    num_students: int = 10,
    question_ids: list = None,
    quality_distribution: dict = None
):
    """
    Crea una struttura di directory con esami fittizi.
    
    Args:
        output_dir: Directory di output
        num_students: Numero di studenti da simulare
        question_ids: Lista di ID domande (None = usa tutte dal QuestionsStore)
        quality_distribution: Distribuzione qualità risposte
    """
    
    # Carica le domande
    try:
        questions_store = QuestionsStore()
        if question_ids is None:
            questions = questions_store.questions[:3]  # Prime 5 domande
        else:
            questions = [questions_store.question(qid) for qid in question_ids]
    except Exception as e:
        print(f"⚠ Errore nel caricare le domande: {e}")
        print("Creando struttura con domande fittizie...")
        from exam import Question, Category
        questions = [
            Question(category=Category("DesignPatterns"), text="Spiega il pattern MVC", id="SE-1"),
            Question(category=Category("Testing"), text="Cos'è il TDD?", id="SE-2"),
            Question(category=Category("Principles"), text="Spiega SOLID", id="SE-3"),
        ]
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f" Creando struttura esame in: {output_path.absolute()}")
    print(f" Numero studenti: {num_students}")
    print(f" Numero domande: {len(questions)}")
    print()
    
    # Seleziona studenti casuali
    selected_students = random.sample(STUDENT_NAMES, min(num_students, len(STUDENT_NAMES)))
    student_codes = {name: generate_student_code() for name in selected_students}
    
    # Crea directory per ogni domanda
    for q_idx, question in enumerate(questions, 1):
        question_dir = output_path / f"Q{q_idx:02d} - {question.id}"
        question_dir.mkdir(exist_ok=True)
        
        print(f"📋 Domanda {q_idx}: {question.id}")
        print(f"   {question.text[:60]}...")
        
        # Crea sottodirectory per ogni studente
        for student_name in selected_students:
            code = student_codes[student_name]
            student_dir = question_dir / f"{code} - {student_name}"
            student_dir.mkdir(exist_ok=True)
            
            # Crea file con la risposta
            answer_file = student_dir / "Attempt_1_textresponse"
            answer = get_random_answer(quality_distribution)
            
            with open(answer_file, "w", encoding="utf-8") as f:
                f.write(answer)
            
            print(f"   ✓ {student_name} ({code})")
        
        print()
    
    # Crea un file README nella directory
    readme_file = output_path / "README.txt"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(f"""MOCK EXAM SUBMISSIONS
=====================

Questa directory contiene esami fittizi generati automaticamente per testare il sistema.

Struttura:
{num_students} studenti
{len(questions)} domande
{num_students * len(questions)} risposte totali

Domande incluse:
""")
        for q_idx, question in enumerate(questions, 1):
            f.write(f"  Q{q_idx:02d}: {question.id} - {question.text}\n")
        
        f.write(f"\nStudenti:\n")
        for name, code in sorted(student_codes.items()):
            f.write(f"  {code} - {name}\n")
        
        f.write(f"""
Distribuzione qualità risposte (approssimativa):
  15% - Eccellente
  35% - Buono
  30% - Sufficiente
  15% - Insufficiente
   5% - Errato

Per valutare questi esami:
  python -m exam.assess {output_dir}
""")
    
    print(f" Struttura creata con successo!")
    print(f"\n Statistiche:")
    print(f"   Directory: {output_path.absolute()}")
    print(f"   Studenti: {len(selected_students)}")
    print(f"   Domande: {len(questions)}")
    print(f"   Risposte totali: {len(selected_students) * len(questions)}")
    print(f"\n Per valutare gli esami:")
    print(f"   python -m exam.assess {output_dir}")
    
    return output_path

def main():
    """Funzione principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Genera esami fittizi per testing")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="mock_exam_submissions",
        help="Directory di output (default: mock_exam_submissions)"
    )
    parser.add_argument(
        "--students", "-s",
        type=int,
        default=3,
        help="Numero di studenti (default: 10)"
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        nargs="+",
        help="ID delle domande da includere (default: prime 5)"
    )
    parser.add_argument(
        "--excellent", type=int, default=15,
        help="Percentuale risposte eccellenti (default: 15)"
    )
    parser.add_argument(
        "--good", type=int, default=35,
        help="Percentuale risposte buone (default: 35)"
    )
    parser.add_argument(
        "--sufficient", type=int, default=30,
        help="Percentuale risposte sufficienti (default: 30)"
    )
    parser.add_argument(
        "--insufficient", type=int, default=15,
        help="Percentuale risposte insufficienti (default: 15)"
    )
    parser.add_argument(
        "--wrong", type=int, default=5,
        help="Percentuale risposte errate (default: 5)"
    )
    
    args = parser.parse_args()
    
    quality_dist = {
        "excellent": args.excellent,
        "good": args.good,
        "sufficient": args.sufficient,
        "insufficient": args.insufficient,
        "wrong": args.wrong,
    }
    
    create_mock_exam_structure(
        output_dir=args.output,
        num_students=args.students,
        question_ids=args.questions,
        quality_distribution=quality_dist
    )

if __name__ == "__main__":
    main()