from transformers import AutoTokenizer, OPTForCausalLM
import pandas as pd
import numpy as np
from typing import Dict
import nltk


#path_to_model = "/home/chavah/scratch/Llama-2-7B-fp16"
model_to_use = "350m" #"6.7b" #"6.7b" "2.7b" "1.3b" "350m" "125m"
layer = -12
list_of_datasets = ["companies"] #["animals","animals_heb","capitals","capitals_heb","cities_heb","colors","colors_heb","companies","companies_heb","elements","elements_heb","foods_heb","inventions","inventions_heb","movies","movies_heb"]#["uncommon"]#["generated"] #

remove_period = True
model = OPTForCausalLM.from_pretrained("facebook/opt-"+model_to_use)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-"+model_to_use)

dfs: Dict[int, pd.DataFrame] = {}
dfs[layer] = pd.DataFrame()
dfs[layer]['statement'] = pd.Series(dtype=object)
dfs[layer]['label'] = pd.Series(dtype='object')
dfs[layer]['embeddings'] = pd.Series(dtype='object')
dfs[layer]['next_id'] = pd.Series(dtype=float)

counter = 0

for dataset_to_use in list_of_datasets:
    # Read the CSV file
    #elements_test_3_sentences_three_labels.csv
    df = pd.read_csv("C:\\Users\\USER\\Downloads\\shevi_chavi\\datasets\\"+dataset_to_use+'_3_sentences.csv')#.head(1000)
    
    j=0
    for i, row in df.iterrows():
        prompt = row['statement']
                
        tokens = tokenizer.tokenize(prompt)
        basic_prompt = prompt
        if remove_period:
            prompt = prompt.rstrip(". ")
        inputs = tokenizer(prompt, return_tensors="pt")
        print(prompt)

        period_indices = [i for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())) if token == '.']

        sentences = prompt.split('. ') 
        #sentences = nltk.sent_tokenize(basic_prompt)
       
        #outputs = model.generate(**inputs, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)#, max_new_tokens=5, min_new_tokens=1) # return_logits=True, max_length=5, min_length=5, do_sample=True, temperature=0.5, no_repeat_ngram_size=3, top_p=0.92, top_k=10)return_logits=True
        outputs = model.generate(inputs.input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)#, max_new_tokens=5, min_new_tokens=1) # return_logits=True, max_length=5, min_length=5, do_sample=True, temperature=0.5, no_repeat_ngram_size=3, top_p=0.92, top_k=10)return_logits=True
        generate_ids = outputs[0]
        next_id = np.array(generate_ids)[0][-1]

        token_ids = inputs["input_ids"][0]  # Extract the tensor from batch
        second_token = row["token_2"]
        last_hidden_state = outputs.hidden_states[0][layer][0][second_token] #[first_generated_word][layer][batch][input_words_for_first_generated_word_only]#last hidden state of first generated word
        sent = sentences[-1]
        print(sent)
        dfs[layer].at[j,'statement'] = sent
        dfs[layer].at[j,'label'] = row['label_2']
        dfs[layer].at[j,'embeddings'] = [last_hidden_state.numpy().tolist()]
        dfs[layer].at[j, 'next_id'] = next_id
        print("processing: " + str(j) + ", next_token:" + str(next_id))
        j=j+1

        '''counter +=1
        if counter >= 1500:
            break
            '''
    
    dfs[layer].to_csv("datasets\\" + "embeddings_only_2nd_" + dataset_to_use + +"_Smollm_1.7B_" + str(abs(layer)) + "_.csv", index=False)
