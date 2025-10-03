from exam import QuestionsStore

store = QuestionsStore()

# print("Domande trovate:")
# for q in store.questions:
#     print(f"  ID: '{q.id}'")
#     print(f"  Category: '{q.category.name}'")
#     print(f"  Text: {q.text[:50]}...")
#     print()

q= store.question("BuildAutomation-1")
print(q)