from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from typing import Dict

#path_to_model = r"C:\Users\talia\Downloads\Diana\opt-2.7b"
model_to_use = "HuggingFaceTB/SmolLM-1.7B"
device = "cuda"  # or "cpu"
layer = -12
list_of_datasets = ["facts"] #["animals","animals_heb","capitals","capitals_heb","cities_heb","colors","colors_heb","companies","companies_heb","elements","elements_heb","foods_heb","inventions","inventions_heb","movies","movies_heb"]#["uncommon"]#["generated"] #
#"companies","capitals", "elements","animals","facts","inventions"
remove_period = True
tokenizer = AutoTokenizer.from_pretrained(model_to_use)

tokenizer.add_special_tokens({"additional_special_tokens": ["<|no_think|>"]})

model = AutoModelForCausalLM.from_pretrained(model_to_use).to(device)
model.resize_token_embeddings(len(tokenizer))

#model = OPTForCausalLM.from_pretrained(path_to_model)
#tokenizer = AutoTokenizer.from_pretrained(path_to_model)

dfs: Dict[int, pd.DataFrame] = {}
dfs[layer] = pd.DataFrame()
dfs[layer]['statement'] = pd.Series(dtype=object)
dfs[layer]['label'] = pd.Series(dtype='object')
dfs[layer]['embeddings'] = pd.Series(dtype='object')
dfs[layer]['next_id'] = pd.Series(dtype=float)

counter = 0
match = 0
not_match = []
for dataset_to_use in list_of_datasets:
    # Read the CSV file
    #elements_test_3_sentences_three_labels.csv
    df = pd.read_csv("datasets\\" + dataset_to_use + "_3_sentences_smallm_1.7.csv")#.head(1000)
    
    j=0
    for i, row in df.iterrows():
        prompt = row['statement']
        
        tokens = tokenizer.tokenize(prompt)
        
        if remove_period:
            prompt = prompt.rstrip(". ")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        #print(prompt)

        period_indices = [i for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())) if token == '.']

        sentences = row['statement'].split('. ') 
       
        #outputs = model.generate(**inputs, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)#, max_new_tokens=5, min_new_tokens=1) # return_logits=True, max_length=5, min_length=5, do_sample=True, temperature=0.5, no_repeat_ngram_size=3, top_p=0.92, top_k=10)return_logits=True
        outputs = model.generate(**inputs, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)
        generate_ids = outputs[0]
        #next_id = np.array(generate_ids)[0][-1]
        next_id = 4
        last_index = outputs.hidden_states[0][layer][0].shape[0] - 1

        #label_columns = [f"label_{idx}" for idx in range(1, 4)]
        #token_columns = [f"token_{idx}" for idx in range(1, 4)]
        '''for idx  in range(min(len(sentences), len(label_columns))):

            if idx in range(len(period_indices)):
                last_token_idx = period_indices[idx]
            else:
                last_token_idx = 0

            token_ids = inputs["input_ids"][0]  # Extract the tensor from batch

        # Get token by index (e.g., index 2)
         token = tokenizer.convert_ids_to_tokens(token_ids[last_token_idx-1].item())
            token0 = tokens[0]
            token_2 = tokenizer.convert_ids_to_tokens(token_ids[-2].item())
            token_1 = tokenizer.convert_ids_to_tokens(token_ids[-1].item())'''
        
        third_token = int(row["token_3"])
        token_ids = inputs["input_ids"][0]
        token = tokenizer.convert_ids_to_tokens(token_ids[-1].item())
        tooken_3 = tokenizer.convert_ids_to_tokens(token_ids[third_token].item())
        

        last_hidden_state = outputs.hidden_states[0][layer][0][third_token] #[first_generated_word][layer][batch][input_words_for_first_generated_word_only]#last hidden state of first generated word
        last_hidden_state_2 = outputs.hidden_states[0][layer][0][-1] 

        if last_hidden_state.cpu().numpy().tolist() == last_hidden_state_2.cpu().numpy().tolist():
            #print("match")
            match+=1
        else:
            print("not match")
            print(prompt)
            not_match.append(token)
            not_match.append(-1)
            not_match.append(tooken_3)
            not_match.append(third_token)
            print("token: " + token)
            print("third_token: " + tooken_3)
            print(f"third_token idx={third_token}, token={tooken_3}")
            print(f"last_token idx={len(tokens)-1}, token={tokens[-1]}")
        sent = sentences[-1]

            #[first_generated_word][layer][batch][input_words_for_first_generated_word_only]#last hidden state of first generated word
        if sent[-1] not in ['.', '..']:
            sent = sent+'.'
        dfs[layer].at[j,'statement'] = sent
        dfs[layer].at[j, 'last_token'] = token
        dfs[layer].at[j,'label'] = row['label_3']
        dfs[layer].at[j,'embeddings'] = [last_hidden_state.cpu().numpy().tolist()]
        dfs[layer].at[j, 'next_id'] = next_id
        #print("processing: " + str(j) + ", next_token:" + str(next_id))
        j=j+1

    print("Matches: " + str(match) + "/" + str(j))  
    print("Not Matches: " + str(len(not_match)))
    print("Not Match Tokens: " + str(not_match))
    exit()        
    dfs[layer].to_csv("embeddings_3_sentences\\3_sentence_only_3rd_last_token_" +  dataset_to_use +"_Smollm_1.7B_" + str(abs(layer)) + ".csv", index=False)
