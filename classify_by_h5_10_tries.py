import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import ast
import tensorflow 
from tensorflow.keras.models import load_model

def correct_str(str_arr):
    val_to_ret = str_arr.replace("[array(", "").replace("dtype=float32)]", "").replace("\n", "").replace(" ", "").replace("],", "]").replace("[", "").replace("]", "")
    return val_to_ret

#model_to_use = "2.7b"
layer_num_from_end = -12
dataset_names = ["capitals","inventions","elements","animals", "companies","facts"]
#["capitals_3_sentences_label_last"]
#["capitals","inventions","elements","animals", "companies","facts"]

# Load all datasets and concatenate them into a single DataFrame
df_list = []
output_file_path = 'results_3_stns\\sub_regular_training.txt'
with open(output_file_path, 'w') as file:
  for dataset_name in dataset_names:
  #    df_list.append(pd.read_csv(
  #        '/home/elishevez/scratch/LLMDEL/datasets/phi/embeddings_with_labels_' + dataset_name + str(
  #            model_to_use) + '_' + str(abs(layer_num_from_end)) + '_rmv_period.csv'))
  #df = pd.concat(df_list, ignore_index=True)
  
    df=pd.read_csv("embeddings_3_sentences\sub_"+dataset_name+"_split_train.csv", encoding='latin1')
    
    # Parse the embeddings if they are stored as strings
    def parse_embeddings(embedding_str):
        try:
            # Clean up the string format if necessary
            cleaned_str = embedding_str.strip()
            return ast.literal_eval(cleaned_str)
        except ValueError:
            print(f"Error parsing: {embedding_str}")
            return np.nan
    
    df['embeddings'] = df['embeddings'].apply(parse_embeddings)
    df = df.dropna(subset=['embeddings'])  
    embeddings = np.vstack(df['embeddings'].dropna().values)
    labels = df['label'][df['embeddings'].notna()]
    
    num_of_runs = 10
    results = []
    tot_acc = 0
    for j in range(num_of_runs):
    #LOAD THE H5 MODEL
      model = load_model("h5\\classifier_"+dataset_name+"_"+str(j)+"_.h5")
      #classifier_capitals_0_.h5
      
      # Evaluate the model on the set
      loss, accuracy = model.evaluate(embeddings, labels, verbose=0)
      
      # Make predictions on the set
      pred_prob = model.predict(embeddings).flatten()
      pred = (pred_prob > 0.5).astype(int)
      
      
      # Calculate accuracy and AUC
      accuracy = accuracy_score(labels, pred)
      auc = roc_auc_score(labels, pred_prob)
      tot_acc += accuracy
    
    avg_acc = tot_acc/num_of_runs
      # Write the results to a file
    file.write(f'Dataset: {dataset_name}\n')
    file.write(f'Accuracy: {avg_acc}\n')
    file.write(f'AUC: {auc}\n')
   
      
