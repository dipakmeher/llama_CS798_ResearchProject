import argparse
import os
import pandas as pd
import numpy as np
from transformers import pipeline, BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import re
import time

# Command line arguments setup
parser = argparse.ArgumentParser(description="Process datasets and calculate embeddings with Llama Chat and BERT.")
parser.add_argument('--amazon_dir', type=str, required=True, help="Directory path for original CSV files.")
parser.add_argument('--overlapping_dir', type=str, required=True, help="Directory path for storing overlapping CSV files.")
parser.add_argument('--csv_file_1', type=str, required=True, help="Filename of the first original CSV file.")
parser.add_argument('--csv_file_2', type=str, required=True, help="Filename of the second original CSV file.")
parser.add_argument('--pair_name', type=str, required=True, help="Pair name for naming the overlapping CSV files.")
args = parser.parse_args()

# Initialize the Llama Chat model
llama2chat = pipeline(model="meta-llama/Llama-2-7b-chat-hf")

# Initialize BERT tokenizer and model for embedding generation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_user_preference(reviewText, summary):
    """Generate user preference using Llama 2 Chat based on review text and summary."""
    prompt = f"""The user left the following review for a product:'{reviewText}'.
    The product summary describes it as a '{summary}'.
    Based on the review and summary of a product, can you analyze the user's preferences (be as specific as you can)?"""
    
    response = llama2chat(prompt)
    
    generated_text = response[0]['generated_text']
    if "Answer:" in generated_text:
        actual_response = generated_text.split("Answer:")[1]
    else:
        actual_response = generated_text
    actual_response = re.sub(r'\n+', '\n', actual_response).strip()

    return actual_response

def get_bert_embedding(text):
    """Function to generate BERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    embedding = outputs.pooler_output.detach().numpy()

    # Check if the embedding is empty or has NaN values
    if np.isnan(embedding).any():
        print(f"NaN embedding for text: {text}")
    
    return embedding

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate the cosine similarity between two embeddings."""
    if isinstance(embedding1, pd.Series):
        embedding1 = embedding1.iloc[0]
    if isinstance(embedding2, pd.Series):
        embedding2 = embedding2.iloc[0]

    if embedding1.ndim > 1:
        embedding1 = embedding1.squeeze()
    if embedding2.ndim > 1:
        embedding2 = embedding2.squeeze()

    return 1 - cosine(embedding1, embedding2)

def process_dataset(csv_path, tokenizer, bert_model):
    df = pd.read_csv(csv_path)
    selected_columns = ['reviewerID', 'asin', 'reviewText', 'summary', 'overall']
    df_selected = df[selected_columns]
    df_sorted = df_selected.sort_values(by='reviewerID').head(2)

    df_sorted['user_preference'] = df_sorted.apply(lambda row: get_user_preference(row['reviewText'], row['summary']), axis=1)
    df_sorted['embedding'] = df_sorted['user_preference'].apply(get_bert_embedding)
    
    print(df_sorted['user_preference'].isnull().sum())
    print(df_sorted['embedding'].apply(lambda x: np.isnan(x).any() or np.isinf(x).any()).sum())


    return df_sorted

# Main execution logic
def main():
    csv_path_1 = os.path.join(args.amazon_dir, args.csv_file_1)
    csv_path_2 = os.path.join(args.amazon_dir, args.csv_file_2)
    overlapping_csv_1 = os.path.join(args.overlapping_dir, f"{args.csv_file_1.split('.')[0]}_{args.pair_name}.csv")
    overlapping_csv_2 = os.path.join(args.overlapping_dir, f"{args.csv_file_2.split('.')[0]}_{args.pair_name}.csv")

    # Load datasets and find overlapping users
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)
    overlapping_users = pd.Series(list(set(df1['reviewerID']).intersection(set(df2['reviewerID']))))
    df1_overlapping = df1[df1['reviewerID'].isin(overlapping_users)]
    df2_overlapping = df2[df2['reviewerID'].isin(overlapping_users)]

    # Print lengths and unique user counts
    print(f"Length of df1_overlapping: {len(df1_overlapping)}")
    print(f"Number of unique users in df1_overlapping: {df1_overlapping['reviewerID'].nunique()}")
    print(f"Length of df2_overlapping: {len(df2_overlapping)}")
    print(f"Number of unique users in df2_overlapping: {df2_overlapping['reviewerID'].nunique()}")

    df_sorted_1 = process_dataset(csv_path_1, tokenizer, bert_model)
    df_sorted_2 = process_dataset(csv_path_2, tokenizer, bert_model)

    # Save the new dataframes to CSV files
    df1_overlapping.to_csv(overlapping_csv_1, index=False)
    df2_overlapping.to_csv(overlapping_csv_2, index=False)


    # Ensure 'reviewerID' is of the same data type
    df1['reviewerID'] = df1['reviewerID'].astype(str)
    df2['reviewerID'] = df2['reviewerID'].astype(str)


    print(df1['reviewerID'].dtype, df2['reviewerID'].dtype)
    print(set(df1['reviewerID']) == set(df2['reviewerID']))
    print(df1['reviewerID'].head(), df2['reviewerID'].head())


    # Calculate cosine similarities and average cosine similarity
    df_sorted_1.set_index('reviewerID', inplace=True)
    df_sorted_2.set_index('reviewerID', inplace=True)
    
    
    cosine_similarities = []
    #common_uids = set(df_sorted_1.index).intersection(set(df_sorted_2.index))
    #print("Unique IDs in df1:", set(df1['reviewerID']))
    #print("Unique IDs in df2:", set(df2['reviewerID']))
    common_uids = set(df1['reviewerID']).intersection(set(df2['reviewerID']))
    print("Common UIDs:", common_uids)
    print("Number of common UIDs:", len(common_uids))
    
    if not common_uids:
        print("No common UIDs found.")
    else:
        cosine_similarities = [calculate_cosine_similarity(df_sorted_1.loc[uid, 'embedding'], df_sorted_2.loc[uid, 'embedding']) for uid in common_uids if not np.isnan(df_sorted_1.loc[uid, 'embedding']).any() and not np.isnan(df_sorted_2.loc[uid, 'embedding']).any()]
    #cosine_similarities = [calculate_cosine_similarity(df_sorted_1.loc[uid, 'embedding'], df_sorted_2.loc[uid, 'embedding']) for uid in common_uids]
    average_cosine_similarity = np.mean(cosine_similarities)

    print(f"Average Cosine Similarity between {args.csv_file_1.split('.')[0]}_{args.pair_name} and {args.csv_file_2.split('.')[0]}_{args.pair_name}: {average_cosine_similarity}")


if __name__ == "__main__":
    main()

