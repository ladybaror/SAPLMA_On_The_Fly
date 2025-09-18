import pandas as pd
import numpy as np

# קריאת הקובץ המקורי
df = pd.read_csv('C:/Users/chavi/Desktop/LLMdel/datasets/opt_2-7/capitals_split_test.csv', encoding='utf-8')

# המרת העמודה 'embeddings' לרשימת מספרים
df['embeddings'] = df['embeddings'].apply(eval)

# הכנת רשימה לשמירת התוצאות
results = []

# עיבוד כל שורה בקובץ
for index in range(len(df) - 1):  # עד השורה הלפני אחרונה
    # קח את האמבדינגס של השורה הנוכחית והשורה הבאה
    current_embeddings = df.iloc[index]['embeddings']
    next_embeddings = df.iloc[index + 1]['embeddings']

    # שרשור האמבדינגס של שתי השורות
    concatenated_embeddings = current_embeddings + next_embeddings

    # שמירת התוצאה
    results.append({
        'statement': df.iloc[index]['statement'],
        'concatenated_embeddings': concatenated_embeddings,
        'label': df.iloc[index]['label']
    })

# יצירת DataFrame חדש מהתוצאות
new_df = pd.DataFrame(results)

# שמירת הקובץ החדש ל-CSV
new_df.to_csv("C:/Users/chavi/Desktop/LLMdel/datasets/opt_2-7/concat_capitals_split_test.csv", index=False)