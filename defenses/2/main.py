import os
import bm25_response_model as model

# import response_model as model
import tree_builder

decision_tree = tree_builder.build_decision_tree("./KnowledgeBase")
# Открываем файл для чтения
questions_pull = []
with open("questions.txt", "r", encoding="utf-8") as file:
    content = file.readlines()
    for i in content:
        questions_pull.append(i.strip().split("|")[1])

# # print(decision_tree)
#
model = model.ResponseModel(decision_tree)
questions = ["кредитные каникулы"]
answers = model.get_answers(questions_pull)

model_name = "Sentence transformers\n"
with open("answers_test_model_name.txt", "w", encoding="utf-8") as file:
    file.write(model_name)
    count = 0
    for answer in answers:
        count += 1

        file.write(
            f"Test question {count}, similiarity - {answer[1]}, len - {len(answer[0])}\n\n"
        )
        file.write(answer[0][0])
        file.write("\n\nEnd of test question\n")
