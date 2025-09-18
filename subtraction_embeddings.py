import pandas as pd
import numpy as np
dataset_to_use = "animals"  # Example dataset name
layer = -12

df = pd.read_csv("embeddings_3_sentences\\3_sentence_with_tokens_"+dataset_to_use+"_Smollm_1.7B_12.csv", encoding='utf-8')

df['embeddings'] = df['embeddings'].apply(eval)

def subtract_embeddings(group):
    sizes = df.groupby(df.index // 3).size()
    if len(group) < 3:  
        print(f"Group size {len(group)} is less than 3, skipping subtraction.")
        return group    
    new_embeddings = [
        group.iloc[0]['embeddings'],
        np.subtract(group.iloc[1]['embeddings'], group.iloc[0]['embeddings']).tolist(),
        np.subtract(group.iloc[2]['embeddings'], group.iloc[1]['embeddings']).tolist()
    ]
    return pd.DataFrame({
        'statement': group['statement'],
        # 'embeddings_old': group['embeddings'],
        'embeddings': new_embeddings,
        'label': group['label']
    })

# חישוב האמבדינגס המחוסרים לכל שלוש רשומות ויצירת DataFrame חדש
new_df = df.groupby(df.index // 3).apply(subtract_embeddings).reset_index(drop=True)

print(new_df.head(1).to_string())

# שמירת הקובץ החדש
new_df.to_csv("embeddings_3_sentences\\sub_"+dataset_to_use+"_split_train.csv", index=False)