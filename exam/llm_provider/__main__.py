from exam.llm_provider import llm_client


def main():
    llm,_,_ = llm_client()
    result = llm.invoke("Quanti gradi ci sono domani settimana prossima Granada più o meno, fammi un stima ?? oggi è il 18/11/25 n")
    print(result)

















if __name__ == "__main__":
    main()