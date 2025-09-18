import pandas as pd

# Read the CSV file
df = pd.read_csv("resources\\elements.csv")

has_the_atomic_number = " has the atomic number of "
dataset = []
#AtomicNumber	Symbol	Name	StandardState	MeltingPoint	GroupBlock	YearDiscovered	info


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
    name = row["Name"]
    atomic_number = row["AtomicNumber"]
    symbol = row["Symbol"]
    standard_state = row["StandardState"]
    #melting_point = row["MeltingPoint"]
    group_block = row["GroupBlock"]
    #year_discovered = row["YearDiscovered"]
    info = row["info"]

    # Atomic number
    is_element = "  is an element."
    '''has_the_atomic_number = " It has the atomic number of "
    sentence = name + is_element + has_the_atomic_number + str(atomic_number) + "."
    dataset.append((sentence, 1))'''

    #sentence = name + has_the_atomic_number + get_other("AtomicNumber", atomic_number) + ."
    #dataset.append((sentence, 0))

    # Symbol
    has_the_symbol = "It has the symbol "
    sentence = name + is_element+ has_the_symbol + str(symbol) + "."
    
    #sentence = name + + has_the_atomic_number + get_other("AtomicNumber", atomic_number) + ."
    dataset.append((sentence, 1))

    sentence = name + is_element + has_the_symbol + get_other("Symbol", symbol) + "."
    dataset.append((sentence, 0))

    # Standard state
    ''' has_the_standard_state = " appears in its standard state as "
    sentence = name + has_the_standard_state + str(standard_state) + "."
    dataset.append((sentence, 1))

    sentence = name + has_the_standard_state + get_other("StandardState", standard_state) + "."
    dataset.append((sentence, 0))
    '''
    # # Melting point
    # has_the_melting_point = " has a melting point of "
    # sentence = name + has_the_melting_point + str(melting_point) + " K."
    # dataset.append((sentence, 1))
    #
    # sentence = name + has_the_melting_point + get_other("MeltingPoint", melting_point) + " K."
    # dataset.append((sentence, 0))

    # Group block
    '''is_in_the_group_block = " is in the "
    sentence = name + is_in_the_group_block + str(group_block) + " group."
    dataset.append((sentence, 1))

    sentence = name + is_in_the_group_block + get_other("GroupBlock", group_block) + " group."
    dataset.append((sentence, 0))
    
    # # Year discovered
    # sentence = name + " " + get_year_info(year_discovered) + "."
    # dataset.append((sentence, 1))
    #
    # sentence = name + " " + get_year_info(get_other("YearDiscovered", year_discovered)) + "."
    # dataset.append((sentence, 0))

    # Additional info
    sentence = name + " " + str(info) + "."
    dataset.append((sentence, 1))

    sentence = name + " " + get_other("info", info) + "."
    dataset.append((sentence, 0))
    '''


# Shuffle the dataset
import random

random.shuffle(dataset)

# Print the first 10 examples in the dataset
print(dataset[:10])
dataset_df = pd.DataFrame(dataset, columns=["statement", "label"])
dataset_df.to_csv("resources\\elements_it_is.csv", index=False) #, header=False)