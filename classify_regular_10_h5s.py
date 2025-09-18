
from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM
import pandas as pd
import numpy as np
from typing import Dict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import csv

#path_to_model = "/home/management/projects/amosa/dianab/LLModel/OPT-6.7/opt-6.7b/"
model_to_use = "SmolLM-1.7B" #"6.7b" "2.7b" "1.3b" "350m"
layers_to_use = [-12]#[-1, -4, -8, -12, -16]
list_of_datasets = ["capitals", "inventions", "elements", "animals", "companies", "facts"] #["facts"] #["capitals", "inventions", "elements", "animals", "facts", "companies"]#["uncommon"]#["generated"] #, "capitals", "inventions", "elements", "animals", "facts", "companies"]
use_logistic_regression = False
use_random_labels = False
accuracy_on_train = False
use_median_length_as_label = False

def correct_str(str_arr):
    val_to_ret = str_arr.replace("[array(", "").replace("dtype=float32)]", "").replace("\n","").replace(" ","").replace("],","]").replace("[","").replace("]","")
    return val_to_ret
        
        
datasets = []
for dataset_name in list_of_datasets:
    #datasets.append(pd.read_csv('resources\\embeddings_with_labels_'+dataset_name+'6.7b_5fromend_rmv_period.csv'))
    df=pd.read_csv("embeddings/" + "embeddings_no_think_regular_" + dataset_name+'_' + model_to_use + "_" + str(abs(-12)) + "_rmv_period.csv", encoding='latin1')

num_of_runs = 10
results = []
for i in range(len(list_of_datasets)):
    test_df = df[i]
    dfs_to_concatenate = df[:i] + df[i + 1:]
    train_df = pd.concat(dfs_to_concatenate, ignore_index=True)

    tot_acc = 0
    for j in range(num_of_runs):
      
        train_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in train_df['embeddings'].tolist()])
        test_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in test_df['embeddings'].tolist()])
        if use_median_length_as_label:
            # Compute the median sentence length for both training and testing datasets
            median_train_length = np.median(train_df['statement'].str.len())
            median_test_length = np.median(test_df['statement'].str.len())
            # Set labels based on the sentence length in relation to the median
            train_df['label'] = np.where(train_df['statement'].str.len() > median_train_length, 1, 0)
            test_df['label'] = np.where(test_df['statement'].str.len() > median_test_length, 1, 0)
        train_labels = np.array(train_df['label'])
        test_labels = np.array(test_df['label'])

        if use_random_labels:
            # Generate random labels of 0 and 1
            train_labels = np.random.randint(2, size=len(train_df))
            test_labels = np.random.randint(2, size=len(test_df))


        # Define the neural network model
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=train_embeddings.shape[1])) #change input_dim to match the number of elements in train_embeddings...
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        if use_logistic_regression:
            model = Sequential()
            model.add(Dense(1, activation='sigmoid', input_dim=train_embeddings.shape[1]))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile the model
        #from tensorflow.keras.models import load_model
        #model = load_model("resources\\classifier_all.h5")

        # Train the model
    
        model.fit(train_embeddings, train_labels, epochs=5, batch_size=32, validation_data=(test_embeddings, test_labels))
        loss, accuracy = model.evaluate(test_embeddings, test_labels)
        if accuracy_on_train:
            loss, accuracy = model.evaluate(train_embeddings, train_labels)
        tot_acc += accuracy
        model.save("h5/classifier_"+str(j)+"_"+list_of_datasets[i]+".h5")
    results.append((list_of_datasets[i], tot_acc/num_of_runs))
    

print(results)
'''with open('results/results_regular.csv','w') as f:
    w = csv.writer(f)
   
    w.writerows(results)
    # Extract the second item from each tuple and put it in a list
    acc_list = [t[1] for t in results]
    # Calculate the average of the numbers in the list
    avg_acc = sum(acc_list) / len(acc_list)
    print(avg_acc)
'''

