import pandas as pd
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import random


class DatasetCreator:
    def __init__(self, tokenizer_name, max_length=512, embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initializes the DatasetCreator class.
        
        Parameters:
        - tokenizer_name: The name of the tokenizer (e.g., "gpt2", "bert-base-uncased")
        - max_length: Maximum length of the input sequence
        - embedding_model_name: The name of the embedding model for sentence transformers
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.misconception_similarity_matrix = None

    def create_input_text(self, question, correct_answer, incorrect_answer, construct_name=None, subject_name=None, special_formatting=False):
        """
        Creates input text based on the question, correct answer, incorrect answer, construct name, and subject name.
        
        Parameters:
        - question: The question text
        - correct_answer: The correct answer text
        - incorrect_answer: The incorrect answer text
        - construct_name: The name of the construct (optional)
        - subject_name: The name of the subject (optional)
        - special_formatting: Whether to use special formatting for the prompt (optional)
        
        Returns:
        - input_text: Formatted text ready for tokenization
        """
        if special_formatting:
            input_text = (
                f"Here is a question for you:\n"
                f"{question}\n"
                f"The correct answer is: {correct_answer}\n"
                f"However, a common misconception is: {incorrect_answer}\n"
            )
            if construct_name:
                input_text += f"This question is related to: {construct_name}.\n"
            if subject_name:
                input_text += f"Subject area: {subject_name}."
        else:
            messages = [
                f"Question: {question}",
                f"Correct Answer: {correct_answer}",
                f"Incorrect Answer: {incorrect_answer}"
            ]
            
            if construct_name:
                messages.append(f"Construct Name: {construct_name}")
            if subject_name:
                messages.append(f"Subject Name: {subject_name}")
            
            input_text = "\n".join(messages)
        
        return input_text
    
    def tokenize_data(self, input_text):
        """
        Tokenizes the input text using the provided tokenizer.
        
        Parameters:
        - input_text: The input text to be tokenized
        
        Returns:
        - tokenized_data: Tokenized representation of the input text
        """
        return self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
    
    def create_similarity_matrix(self, misconception_mapping, embedding_model_name=None):
        """
        Creates a similarity matrix of all the misconception names using sentence embeddings.
        
        Parameters:
        - misconception_mapping: A DataFrame containing the Misconception IDs and their names
        - embedding_model_name: The name of the embedding model (optional)
        
        Returns:
        - similarity_matrix: A matrix representing similarity between all misconception names
        """
        if embedding_model_name:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        
        misconception_names = misconception_mapping['MisconceptionName'].tolist()
        embeddings = self.embedding_model.encode(misconception_names, convert_to_tensor=True, show_progress_bar=True)
        #similarity_matrix = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        similarity_matrix = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1).cpu(), embeddings.unsqueeze(0).cpu(), dim=-1)
        self.misconception_similarity_matrix = similarity_matrix
        
        return similarity_matrix
    
    def create_dataset(self, df, misconception_mapping, include_construct_name=False, include_subject_name=False, special_formatting=False, scoring_type='binary', return_type='tokenized', testing=False):
        """
        Creates a dataset for direct preference optimization (DPO).
        
        Parameters:
        - df: The training DataFrame containing all the necessary columns
        - misconception_mapping: A DataFrame containing the Misconception IDs and their names
        - include_construct_name: Whether to include the construct name in the prompt (optional)
        - include_subject_name: Whether to include the subject name in the prompt (optional)
        - special_formatting: Whether to use special formatting for the prompts (optional)
        - scoring_type: The type of scoring ('binary' or 'graded')
        - return_type: The type of return ('tokenized' or 'text')
        
        Returns:
        - A dictionary containing either tokenized input tensors or text for the DPO model
        """
        inputs, labels, rejected_responses, chosen_scores, rejected_scores = [], [], [], [], []
        misconception_dict = pd.Series(misconception_mapping.MisconceptionName.values, index=misconception_mapping.MisconceptionId).to_dict()
        
        if testing:
            df = df.head(3)
        
        for index, row in df.iterrows():
            question = row['QuestionText']
            correct_answer = row[f'Answer{row.CorrectAnswer}Text']
            
            for answer_label in ['A', 'B', 'C', 'D']:
                misconception_id = row.get(f'Misconception{answer_label}Id', None)
                if pd.notna(misconception_id):
                    incorrect_answer = row[f'Answer{answer_label}Text']
                    misconception_name = misconception_dict.get(misconception_id, "Unknown Misconception")
                    
                    construct_name = row['ConstructName'] if include_construct_name and 'ConstructName' in df.columns else None
                    subject_name = row['SubjectName'] if include_subject_name and 'SubjectName' in df.columns else None
                    
                    # Create the input text
                    input_text = self.create_input_text(
                        question, correct_answer, incorrect_answer, 
                        construct_name, subject_name, special_formatting
                    )
                    
                    # Determine scoring
                    if scoring_type == 'binary':
                        least_similar_idx = torch.argmin(self.misconception_similarity_matrix[int(misconception_id)]).item()
                        chosen_score = 1
                        rejected_score = 0
                        rejected_response = misconception_dict[least_similar_idx]
                    elif scoring_type == 'graded':
                        chosen_score = 5
                        random_idx = random.choice(range(len(self.misconception_similarity_matrix)))
                        similarity = self.misconception_similarity_matrix[int(misconception_id), int(random_idx)].item()
                        rejected_score = max(0, similarity * 5)  # Sharpened similarity scaled by 5
                        rejected_response = misconception_dict[random_idx]
                    else:
                        raise ValueError("Invalid scoring type. Choose either 'binary' or 'graded'.")
                    
                    chosen_scores.append(chosen_score)
                    rejected_scores.append(rejected_score)
                    
                    if return_type == 'tokenized':
                        tokenized_data = self.tokenize_data(input_text)
                        inputs.append(tokenized_data['input_ids'])
                        labels.append(self.tokenize_data(misconception_name)['input_ids'])
                        rejected_responses.append(self.tokenize_data(rejected_response)['input_ids'])
                    elif return_type == 'text':
                        inputs.append(input_text)
                        labels.append(misconception_name)
                        rejected_responses.append(rejected_response)
                    else:
                        raise ValueError("Invalid return type. Choose either 'tokenized' or 'text'.")

                    if testing:
                        print(f"{input_text}")
                        print(f"chosen_response: {misconception_name} Chosen Score: {chosen_score}, Rejected_response: {rejected_response} ,Rejected Score: {rejected_score}\n\n")
        
        if return_type == 'tokenized':
            return {
                'instruction': torch.cat(inputs, dim=0),
                'chosen_response': torch.cat(labels, dim=0),
                'rejected_response': torch.cat(rejected_responses, dim=0),
                'chosen_score': torch.tensor(chosen_scores, dtype=torch.float32),
                'rejected_score': torch.tensor(rejected_scores, dtype=torch.float32)
            }
        elif return_type == 'text':
            return {
                'instruction': inputs,
                'chosen_response': labels,
                'rejected_response': rejected_responses,
                'chosen_score': chosen_scores,
                'rejected_score': rejected_scores
            }



if __name__ == "__main__":
    misconception = pd.read_csv("data/misconception.csv")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    tokenizer_name = "Qwen/Qwen2.5-0.5B"
    embedding_model_name = 'all-MiniLM-L6-v2'
    dataset_creator = DatasetCreator(tokenizer_name, embedding_model_name=embedding_model_name)

    dataset_creator.create_similarity_matrix(misconception)

    dataset = dataset_creator.create_dataset(train, misconception, include_construct_name=True, include_subject_name=True, special_formatting=True, scoring_type='binary', return_type='text', testing=False)

    #save
    torch.save(dataset, 'data/dataset.pt')

