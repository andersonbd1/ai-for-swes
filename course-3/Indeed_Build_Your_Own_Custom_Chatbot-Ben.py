# pip install -q openai==0.27.7 tiktoken openai[datalib] numpy==1.23.5 pandas matplotlib plotly scipy scikit-learn tenacity numpy
# I switched from a notebook to a script so that I could more easily debug it.

import openai
import pandas as pd
import numpy as np
import tiktoken
from openai.embeddings_utils import distances_from_embeddings

# TODO: write an explanation of which dataset you have chosen and why it is appropriate for this task
# I downloaded the entire Catholic Bible from here:
# https://huggingface.co/datasets/SzuTao/CatholicBible
# I then limited it to the Gospel of John:
# $ cat ./data/CatholicBible.csv | grep '^John' > ./data/CatholicBible-John.csv
# I did this so that I could have the chatbot answer particular questions about the Gospel of John by
# citing the chapter and verse.  I think it worked pretty well, but so did openai on its own without
# the help of my custom embeddings :-)... but I think the exercise still proved worthy.

openai.api_base = "https://openai.vocareum.com/v1"

# This function is complete and should not be modified.
def get_openai_api_key():
    #return 'REDACTED'
    return 'voc-62161097126677383455067ac76d27f0ad0.64235909'

openai.api_key = get_openai_api_key()

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    # Prior to running this script, I combined the columns into one ('verse') containing the whole thing:
    # [Book] [Chapter]:[Verse], [Text of verse]
    df['text'] = df['verse'] # use
    return df[['text']]

def generate_embeddings(df, embedding_model_name="text-embedding-ada-002", batch_size=1):
    embeddings = []
    for i in range(0, len(df), batch_size):
        response = openai.Embedding.create(
            input=df.iloc[i:i + batch_size]["text"].tolist(),
            engine=embedding_model_name
        )
        embeddings.extend([data["embedding"] for data in response["data"]])
        # just to track the progress
        print(f"in generate_embeddings {i}")
    df["embeddings"] = embeddings
    return df

def save_embeddings(df, output_file):
    df.to_csv(output_file, index=False)

def load_embeddings(file_path):
    df = pd.read_csv(file_path)
    df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)
    return df

def get_relevant_rows(question, df, embedding_model_name="text-embedding-ada-002", top_n=10):
    question_embedding = openai.Embedding.create(
        model=embedding_model_name,
        input=question
    )['data'][0]['embedding']

    df_copy = df.copy()
    df_copy['distance'] = distances_from_embeddings(question_embedding, df_copy['embeddings'].values, distance_metric="cosine")

    return df_copy.nsmallest(top_n, 'distance')

# ===============================
# Prompt Creation & Answering
# ===============================

def create_prompt(question, df, max_token_count=1500):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Answer the question based on the context below. If the question can't be answered based on the context, say "I don't know."
    prompt_template = """
    Answer the question based on the context below.

    Context: {}

    ---

    Question: {}

    Answer:
    """
    current_token_count = len(tokenizer.encode(prompt_template)) + len(tokenizer.encode(question))

    context = []
    for text in df["text"].values:
        tokens_in_text = len(tokenizer.encode(text))
        if current_token_count + tokens_in_text <= max_token_count:
            context.append(text)
            current_token_count += tokens_in_text
        else:
            break

    return prompt_template.format("\n\n###\n\n".join(context), question)

def get_openai_answer(prompt, max_answer_tokens=150):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=max_answer_tokens
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error: {str(e)}")
        return "An error occurred."

# ===============================
# Question Answering Functions
# ===============================

def answer_basic_question(question, max_answer_tokens=150):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=question,
            max_tokens=max_answer_tokens
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error: {str(e)}")
        return "An error occurred."

# def answer_question_with_context(question, df, max_prompt_tokens=1500, max_answer_tokens=150, top_n=5):
def answer_question_with_context(question, df, max_prompt_tokens=5000, max_answer_tokens=150, top_n=100):
    relevant_rows = get_relevant_rows(question, df, top_n=top_n)
    prompt = create_prompt(question, relevant_rows, max_token_count=max_prompt_tokens)

    return get_openai_answer(prompt, max_answer_tokens=max_answer_tokens)

# ===============================
# Main Function
# ===============================

def main():
    # df = load_dataset("./data/CatholicBible-John.csv")

    # df = generate_embeddings(df)
    # save_embeddings(df, "./data/embeddings_with_vectors-John.csv")

    df = load_embeddings("data/embeddings_with_vectors-John.csv")

    # Example Question 1
    for q in [
        "Does anyone love me or care about me?",
        "Why should I go on living?",
    ]:
        basic_answer = answer_basic_question(q)
        custom_answer = answer_question_with_context(q, df)
        print(f"Question: {q}\nBasic Answer: {basic_answer}\n\nCustom Answer: {custom_answer}\n\n")

# ===============================
# Execution
# ===============================
if __name__ == "__main__":
    main()
