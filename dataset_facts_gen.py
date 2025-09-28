import pandas as pd
import random

# Read the CSV file
df = pd.read_csv("resources\\facts_true_false.csv")

#IBM is a name of company. It has headquarter in Japan ...
dataset = []
#Name	Headquarters	Founded	Industry	Revenue	Employees


# Loop through each row in the dataframe
def get_year_info(year_discovered):
    if year_discovered == "Ancient":
        return "was discovered over a thousand years ago"
    return "was discovered in " + str(year_discovered)


def get_other(required_param, current_val):
    other_val = current_val
    while other_val == current_val:  # keep sampling a color until you get somthing different
        other_row = df.sample(n=1)
        other_val = other_row[required_param].values[0]
    return str(other_val)


for i, row in df.iterrows():
    # Extract the country and capital city from the row
    statement = row["statement"]
    label = row["label"]
   

    # Country
    falsehood = " The previous statement is false."
    truth = " The previous statement is a true fact."
    # Randomly return either 0 or 1
    rand = random.choice([0, 1])

    if rand == 0:
        sentence = statement + falsehood
        label = 1 - label  # flip the label
    else:
        sentence = statement + truth
    dataset.append((sentence, label))

    


# Shuffle the dataset
import random

random.shuffle(dataset)

# Print the first 10 examples in the dataset
print(dataset[:10])
dataset_df = pd.DataFrame(dataset, columns=["statement", "label"])
dataset_df.to_csv("resources\\facts_it_is_pairs_amos.csv", index=False) #, header=False)