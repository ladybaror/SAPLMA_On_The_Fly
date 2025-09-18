from transformers import AutoTokenizer, OPTForCausalLM
import pandas as pd
import numpy as np
from typing import Dict
import nltk

#path_to_model = "/home/chavah/scratch/Llama-2-7B-fp16"
#model_to_use = "350m" #"6.7b" #"6.7b" "2.7b" "1.3b" "350m" "125m"
layers_to_use = [ -12]
list_of_datasets = ["capitals"]#,"inventions","elements","animals", "companies","facts"] #["animals","animals_heb","capitals","capitals_heb","cities_heb","colors","colors_heb","companies","companies_heb","elements","elements_heb","foods_heb","inventions","inventions_heb","movies","movies_heb"]#["uncommon"]#["generated"] #



remove_period = True

dfs: Dict[int, pd.DataFrame] = {}

counter = 0

df2 = pd.read_csv("embeddings_3_sentences\\3_sentence_only_3rd_facts_Smollm_1.7B_12.csv", encoding='utf-8')
    # Read the CSV file
df1 = pd.read_csv("embeddings_3_sentences\\3_sentence_with_tokens_facts_Smollm_1.7B_12.csv")#.head(1000)



i=0
count = 0
match = 0
for i, row1 in df1.iterrows():
    count = count+1
    #sentences = row1['statement'].split(('. '))
    #sentences = nltk.sent_tokenize( row1['statement'])

    #print(sentences)
    
    print(row1['statement'])
    
    for j, row2 in df2.iterrows():
        #print(row2['statement'])

        if row1['statement'] == row2['statement']:
            print(row1['statement'])
            if(row1['embeddings'] == row2['embeddings']):
                match = match +1
                print("match")
                continue

            else:
                print("not match")
        
            
print("Matches: "+ str(match) +  "/"+ str(count))




