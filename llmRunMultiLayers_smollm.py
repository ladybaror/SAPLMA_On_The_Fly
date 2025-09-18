from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from typing import Dict
import torch
import chardet


# ----------------------------
# Settings
# ----------------------------
#model_name = "HuggingFaceTB/SmolLM-1.7B"
model_name = "HuggingFaceTB/SmolLM-1.7B"
device = "cuda"  # or "cpu"

#HuggingFaceTB/SmolLM-1.7B

layers_to_use = [-12]
list_of_datasets = ["facts","companies","capitals", "elements","animals","facts","inventions"]
#["facts","companies","capitals", "elements","animals","facts","inventions"]
remove_period = True
max_rows = 1500

# ----------------------------
# Load model + tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.add_special_tokens({"additional_special_tokens": ["<|no_think|>"]})

model = AutoModelForCausalLM.from_pretrained(  model_name).to(device)
model.resize_token_embeddings(len(tokenizer))


print(tokenizer.special_tokens_map)
# ----------------------------
# Storage for results
# ----------------------------
dfs: Dict[int, pd.DataFrame] = {}
counter = 0


custom_template = """{{- bos_token }}
{% for m in messages %}
{{ m['role']|capitalize }}: {{ m['content'] }}
{% endfor %}{{ eos_token }}"""


# ----------------------------
# Process each dataset
# ----------------------------
for dataset_to_use in list_of_datasets:
    # Read the CSV file
    df = pd.read_csv( "resources\\"+dataset_to_use+"_it_is_pairs.csv")#.head(1000)
    df['embeddings'] = pd.Series(dtype='object')
    df['next_id'] = pd.Series(dtype=float)
    for layer in layers_to_use:
        dfs[layer] = df.copy()
        # prepare a copy per layer
       

    for i, row in df.iterrows():
        prompt = row['statement']
        if remove_period:
            prompt = prompt.rstrip(". ")

        # build chat message
        ''' messages_think = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": prompt}
        ]

        # use chat template
        text = tokenizer.apply_chat_template(
            messages_think,
            tokenize=False,
            add_generation_prompt=True,
        )'''

        # tokenize
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        tokens = tokenizer.tokenize(prompt)


        # generate with hidden states
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=1,
            min_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        # get the generated token id
        generate_ids = outputs.sequences
        next_id = np.array(generate_ids.cpu())[0][-1]

        # outputs.hidden_states is a tuple of length [num_layers] per step
        # we want hidden state of first generated token
        # outputs.hidden_states: List[step][layer][batch, seq, hidden_dim]
        # first generated token is step 0 (since max_new_tokens=1)
        hidden_states_for_step = outputs.hidden_states[0]
        print(f"last_token idx={len(tokens)-1}, token={tokens[-1]}")


        for layer in layers_to_use:
            # negative index from end
            emb_tensor = hidden_states_for_step[layer][0, -1, :].detach().cpu().numpy()
            dfs[layer].at[i, 'embeddings'] = [emb_tensor.tolist()]
            dfs[layer].at[i, 'next_id'] = next_id

        #print(f"processing: {i}, next_token: {next_id}")
    

    # save per layer
    for layer in layers_to_use:
        out_path = (
            "embeddings_3_sentences\\is_it_"
            + dataset_to_use
            + "_pairs_SmolLM1.7B_"
            + str(abs(layer))
            + "_rmv_period.csv"
        )
        dfs[layer].to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
