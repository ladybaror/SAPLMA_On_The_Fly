import numpy as np
from transformers import AutoTokenizer
import csv
import pandas as pd
import ast

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# Read the CSV file containing the texts and embeddings
texts_df = pd.read_csv("resources_SAPLMA_on_the_fly/embeddings_with_labels_test_3_sentences_three_labels_animals350m_12_rmv_period.csv", dtype=str, low_memory=False)

# Extract texts, embeddings, and labels
texts = texts_df['statement'].tolist()
embeddings_list = texts_df['embeddings'].tolist()  # This should be a list of strings
labels = texts_df['label_1'].tolist()

# Ensure embeddings are in the correct format (convert from string if needed)
embeddings_matrix_list = []
for embedding_str in embeddings_list:
    try:
        embedding_values = ast.literal_eval(embedding_str)
        embeddings_matrix = np.array(embedding_values)
        embeddings_matrix_list.append(embeddings_matrix)
    except Exception as e:
        print(f"Error processing embedding: {e}")

# Open the CSV file for writing the extracted vectors
with open("resources_SAPLMA_on_the_fly/animals_test_3_sentences_three_labels.csv", "w", newline='', encoding='latin1') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['statement', 'label', 'embeddings'])

    for text, embeddings_matrix, label in zip(texts, embeddings_matrix_list, labels):
        # Ensure the text is a string
        if not isinstance(text, str):
            text = str(text)

        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Find the indices of the last token in each sentence (just before the period or question mark)
        sentence_end_indices = [i for i, token in enumerate(tokens) if token in ['.', '?']]
        print(sentence_end_indices)
        sentences = text.split('. ')  # Split text into sentences

        # Calculate the start index for the last third
        num_sentences = len(sentence_end_indices)
        if num_sentences == 0:
            continue
        third_length = num_sentences // 3
        start_idx = 2 * third_length

        # Extract the corresponding vectors from the last third of the embeddings matrix
        extracted_vectors = []
        for idx in range(start_idx, num_sentences):
            if idx < len(sentences) and sentence_end_indices[idx] < embeddings_matrix.shape[0]:
                extracted_vectors.append((sentences[idx].strip(), label, embeddings_matrix[sentence_end_indices[idx]]))
            else:
                print(f"Index {sentence_end_indices[idx] if idx < len(sentence_end_indices) else 'N/A'} is out of bounds for embeddings with length {embeddings_matrix.shape[0]}")

        # Convert each vector to a string and save each vector as a single cell in a CSV file
        for sentence, label, vector in extracted_vectors:
            vector_str = '[' + ', '.join(map(str, vector)) + ']'
            writer.writerow([sentence, label, vector_str])
