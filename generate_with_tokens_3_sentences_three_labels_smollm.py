import pandas as pd
from transformers import AutoTokenizer, OPTForCausalLM

import csv 
print("Diana")

model_name = "HuggingFaceTB/SmolLM-1.7B"
dataset = "inventions"
# Load the CSV file
data = pd.read_csv("resources//"+dataset+'_true_false.csv', encoding='latin-1')

data = data.sample(frac=1)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.add_special_tokens({"additional_special_tokens": ["<|no_think|>"]})# Create an empty list to store the triplets
triplets = []

# Iterate over the rows of the DataFrame
for i in range(0, len(data), 3):
    # Get the current batch of 3 statements and labels
    batch_statements = data.iloc[i:i + 3, 0].tolist()
    
    all_statements = [s.strip() + '.' if not s.strip().endswith('.') else s.strip() for s in batch_statements]
    triplet_statement = ' '.join(all_statements)

    #all_tokens = tokenizer(triplet_statement, return_tensors="pt", add_special_tokens=False)
    #input_ids_all = all_tokens.input_ids[0]
    prompt = triplet_statement.rstrip(". ")
    print("PROMPT")
    print(prompt)




    token_ids = tokenizer(triplet_statement, return_tensors="pt")
    #decoded_tokens_all = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_tokens_all = tokenizer.convert_ids_to_tokens(token_ids["input_ids"][0].tolist())
    #decoded_tokens_all = tokenizer.convert_ids_to_tokens(input_ids_all)

   
    idx = 0
    tokens_idx = []
    for batch in batch_statements:
        #tokens = tokenizer(batch, return_tensors="pt", add_special_tokens=False)

        # Get tokenized IDs and corresponding words
        #input_ids = tokens.input_ids[0]
        #decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        input_ids = tokenizer(batch,  return_tensors="pt")
        #decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids["input_ids"][0].tolist())


        # Find the last token (excluding special tokens)
        last_token = decoded_tokens[-1]
        print(last_token)

       
        last_token_idx = len(decoded_tokens) - 2

        idx_curr = decoded_tokens_all.index(decoded_tokens[last_token_idx], idx)
        tokens_idx.append(idx_curr)
        idx += last_token_idx
        print("decoded_tokens[last_token_idx]")
        print(decoded_tokens[last_token_idx])
        print("idx_curr")
        print(idx_curr)
        print(decoded_tokens_all[idx_curr])

        
        #print(decoded_tokens_all[idx])
        
        
        
        
        

   
    # Ensure each statement ends with a period
    batch_statements = [s.strip() + '.' if not s.strip().endswith(('.', '..')) else s.strip() for s in batch_statements]
    

       
    batch_labels = data.iloc[i:i + 3, 1].tolist()

    # Add the original order
    triplet_statement = ' '.join(batch_statements)
    triplet_labels = batch_labels
    triplets.append([triplet_statement, *triplet_labels,  *tokens_idx])

# Create a new DataFrame with the triplets
columns = ['statement', 'label_1', 'label_2', 'label_3','token_1', 'token_2', 'token_3']
triplets_df = pd.DataFrame(triplets, columns=columns)

    # Save the new DataFrame to a CSV file
triplets_df.to_csv('datasets//'+dataset+'_3_sentences_smallm_1.7.csv', index=False)
