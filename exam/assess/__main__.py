from exam.assess import *
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Valuta automaticamente le risposte degli studenti',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""


Punteggio:
  - Basato su formula pesata: 70% Core + 20% Important + 10% Additional
  - Mostrato per ogni domanda + totale finale
        """
    )
    
    parser.add_argument(
        'exam_dir',
        type=str,
        help='Directory contenente le risposte degli studenti (struttura: Q01 - ID/CODE - NAME/Attempt_*)'
    )
    
    parser.add_argument(
        '--show-all', '-a',
        action='store_true',
        help='Mostra tutte le valutazioni (CORE + Details), non solo CORE mancanti (default: solo CORE mancanti)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Modello LLM da usare (default: mistralai/mistral-7b-instruct)'
    )
    
    parser.add_argument(
        '--provider', '-p',
        type=str,
        default=None,
        help='Provider del modello (default: openrouter)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostra informazioni dettagliate durante la valutazione'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.exam_dir:
        print("Errore: Devi specificare la directory degli esami")
        print("Uso: python -m exam.assess <exam_directory>")
        sys.exit(1)
    
    # Mostra configurazione
    print("  EXAM ASSESSMENT SYSTEM - New Scoring Model")
    print(f"Exam directory: {args.exam_dir}")
    print(f"Model: {args.model or 'mistralai/mistral-7b-instruct (default)'}")
    print(f"Display mode: {'All features (Core + Details)' if args.show_all else 'Core missing only'}")
    print(f"Scoring: 70% Core + 20% Important Details + 10% Additional Details")
    if OUTPUT_FILE != sys.stdout:
        print(f"Output file: {OUTPUT_FILE.name}")
    print()
    
    try:
        assessor = Assessor(
            args.exam_dir,
            model_name=args.model,
            model_provider=args.provider
        )
    except Exception as e:
        print(f"Errore nell'inizializzazione: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print("Inizio valutazione...")
    print()
    
    try:
        # show_only_missing_core = True per default (mostra solo CORE mancanti)
        # False se --show-all è specificato
        show_only_missing_core = not args.show_all
        
        assessments = assessor.assess_all(show_only_missing_core=show_only_missing_core)
        
        # Stampa riepilogo finale
        print()
        print("  VALUTAZIONE COMPLETATA")
        
        assessments.pretty_print(file=OUTPUT_FILE, show_only_missing_core=show_only_missing_core)
        
        if OUTPUT_FILE is not sys.stdout:
            assessments.pretty_print(file=sys.stdout, show_only_missing_core=show_only_missing_core)
            OUTPUT_FILE.close()
            print(f"\nRisultati salvati in: {OUTPUT_FILE.name}")
        
        # Statistiche finali
        print()
        print("  STATISTICHE FINALI")
        
        all_students = assessments.assessments
        if all_students:
            scores = [s.total_score() for s in all_students]
            max_possible = sum(q.weight for q in all_students[0].answers.keys()) if all_students[0].answers else 0
            
            print(f" valutati: {len(all_students)}")
            print(f"Punteggio medio: {sum(scores)/len(scores):.2f}/{max_possible}")
            print(f"Punteggio massimo: {max(scores):.2f}/{max_possible}")
            print(f"Punteggio minimo: {min(scores):.2f}/{max_possible}")
            print()
        
        print("Valutazione completata con successo!")
        
    except KeyboardInterrupt:
        print("\n\nValutazione interrotta dall'utente")
        if OUTPUT_FILE is not sys.stdout:
            OUTPUT_FILE.close()
        sys.exit(1)
    except Exception as e:
        print(f"\nErrore durante la valutazione: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        if OUTPUT_FILE is not sys.stdout:
            OUTPUT_FILE.close()
        sys.exit(1)


if __name__ == "__main__":
    main()