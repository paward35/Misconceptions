import os
from transformers import AutoTokenizer
import pandas as pd



def prepare_features(df_questions):
    tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")

    PROMPT  = """Question: {Question}
    Incorrect Answer: {IncorrectAnswer}
    Correct Answer: {CorrectAnswer}
    Construct Name: {ConstructName}
    Subject Name: {SubjectName}
    
    Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>.
    Before answering the question think step by step concisely in 1-2 sentence inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag."""
    
    def apply_template(row, tokenizer, targetCol):
        messages = [
            {
                "role": "user", 
                "content": PROMPT.format(
                     ConstructName=row["ConstructName"],
                     SubjectName=row["SubjectName"],
                     Question=row["QuestionText"],
                     IncorrectAnswer=row[f"Answer{targetCol}Text"],
                     CorrectAnswer=row[f"Answer{row.CorrectAnswer}Text"])
            }
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text
    
    df_features = {}
    df_label = {}
    for idx, row in df_questions.iterrows():
        for option in ["A", "B", "C", "D"]:
            if (row.CorrectAnswer!=option) & (row[f"Misconception{option}Id"]!=-1):
                df_features[f"{row.QuestionId}_{option}"] = apply_template(row, tokenizer, option)
                df_label[f"{row.QuestionId}_{option}"] = [row[f"Misconception{option}Id"]]
    df_label = pd.DataFrame([df_label]).T.reset_index()
    df_label.columns = ["QuestionId_Answer", "MisconceptionId"]
    df_label.to_parquet("label.parquet", index=False)
    df_features = pd.DataFrame([df_features]).T.reset_index()
    df_features.columns = ["QuestionId_Answer", "text"]
    df_features.to_parquet("submission.parquet", index=False)
    return df_features, df_label

