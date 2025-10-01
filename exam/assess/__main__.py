from exam.assess import *
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Valuta automaticamente le risposte degli studenti',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:

  # Valutazione standard (solo problemi)
  python -m exam.assess ./mock_exam_submissions

  # Mostra tutte le valutazioni (anche quelle soddisfatte)
  python -m exam.assess ./mock_exam_submissions --show-all

  # Salva output in file
  OUTPUT_FILE=results.txt python -m exam.assess ./mock_exam_submissions

  # Usa modello diverso
  python -m exam.assess ./mock_exam_submissions --model mistralai/mixtral-8x7b-instruct

Modalità di output:
  - Default: Mostra solo feature NON soddisfatte (problemi)
  - --show-all: Mostra tutte le feature (anche quelle OK)

Punteggio:
  - Basato su % di SHOULD soddisfatti
  - Penalità del 15% per ogni SHOULDNT violato
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
        help='Mostra tutte le valutazioni, non solo i problemi (default: solo problemi)'
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
        print(" Errore: Devi specificare la directory degli esami")
        print("Uso: python -m exam.assess <exam_directory>")
        sys.exit(1)
    
    # Mostra configurazione
    print("=" * 70)
    print("  EXAM ASSESSMENT SYSTEM")
    print("=" * 70)
    print(f" Exam directory: {args.exam_dir}")
    print(f" Model: {args.model or 'mistralai/mistral-7b-instruct (default)'}")
    print(f" Display mode: {'All features' if args.show_all else 'Problems only'}")
    if OUTPUT_FILE != sys.stdout:
        print(f"Output file: {OUTPUT_FILE.name}")
    print("=" * 70)
    print()
    
    try:
        assessor = Assessor(
            args.exam_dir,
            model_name=args.model,
            model_provider=args.provider,
            only_critical_features=True  # Sempre solo SHOULD/SHOULDNT
        )
    except Exception as e:
        print(f" Errore nell'inizializzazione: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print(" Inizio valutazione...")
    print()
    
    try:
        # show_only_unsatisfied = True per default (mostra solo problemi)
        # False se --show-all è specificato
        show_only_problems = not args.show_all
        
        assessments = assessor.assess_all(show_only_unsatisfied=show_only_problems)
        
        # Stampa riepilogo finale
        print()
        print("=" * 70)
        print("  VALUTAZIONE COMPLETATA")
        print("=" * 70)
        
        assessments.pretty_print(file=OUTPUT_FILE, show_only_unsatisfied=show_only_problems)
        
        if OUTPUT_FILE is not sys.stdout:
            assessments.pretty_print(file=sys.stdout, show_only_unsatisfied=show_only_problems)
            OUTPUT_FILE.close()
            print(f"\n Risultati salvati in: {OUTPUT_FILE.name}")
        
        # Statistiche finali
        print()
        print("=" * 70)
        print("  STATISTICHE FINALI")
        print("=" * 70)
        
        all_students = assessments.assessments
        if all_students:
            scores = [s.total_score() for s in all_students]
            max_possible = sum(q.weight for q in all_students[0].answers.keys()) if all_students[0].answers else 0
            
            print(f"Studenti valutati: {len(all_students)}")
            print(f" Punteggio medio: {sum(scores)/len(scores):.2f}/{max_possible}")
            print(f" Punteggio massimo: {max(scores):.2f}/{max_possible}")
            print(f" Punteggio minimo: {min(scores):.2f}/{max_possible}")
            print()
        
        print(" Valutazione completata con successo!")
        
    except KeyboardInterrupt:
        print("\n\n  Valutazione interrotta dall'utente")
        if OUTPUT_FILE is not sys.stdout:
            OUTPUT_FILE.close()
        sys.exit(1)
    except Exception as e:
        print(f"\n Errore durante la valutazione: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        if OUTPUT_FILE is not sys.stdout:
            OUTPUT_FILE.close()
        sys.exit(1)


if __name__ == "__main__":
    main()