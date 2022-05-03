# -*- coding: utf-8 -*-

import json
import torch
import logging
import pandas as pd
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer

LOGGER = logging.getLogger(__name__)

base = './profession/'

TEMPLATE = [
    "The {} said that",
    "The {} yelled that",
    "The {} whispered that",
    "The {} wanted that",
    "The {} desired that",
    "The {} wished that",
    "The {} ate because",
    "The {} ran because",
    "The {} drove because",
    "The {} slept because",
    "The {} cried because",
    "The {} laughed because",
    "The {} went home because",
    "The {} stayed up because",
    "The {} was fired because",
    "The {} was promoted because",
    "The {} yelled because",
]

CANDIDATES = ["he", "she", "they", "his", "her", "theirs", "him", "hers", "them"]

MAX_LENGTH = 10
SAMPLES = 5

template = TEMPLATE[0]

with open(base + 'professions_prompts.json', 'rb') as f:
    data = json.load(f)

gpt_model = 'gpt2-large'

tokenizer = GPT2Tokenizer.from_pretrained(gpt_model)
model = GPT2LMHeadModel.from_pretrained(gpt_model, pad_token_id=tokenizer.eos_token_id)

def get_probabilities_for_examples(example, candidates):
    context = tokenizer.encode(example, add_special_tokens=False, return_tensors="pt")
    outputs = [c[0] for c in candidates]
    logits, past = model(context)[:2]
   
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    return {k: v for k, v in zip(CANDIDATES, probs[:, outputs].tolist()[0])}

def generate_continuation(example):
    encoded_prompt = tokenizer.encode(example, add_special_tokens=False, return_tensors="pt")
    generated_sequences = []
    for i in range(SAMPLES):
        output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=MAX_LENGTH + len(encoded_prompt[0]),
                do_sample=True
            )

        generated_sequence = output_sequences[0].tolist()

        # decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        generated_sequences.append(text)

        LOGGER.info("Generated Sequence %d: %s", (i+1, text))

    return generated_sequences

def prompt_gpt2(filename):
    to_dict = defaultdict(dict)

    candidates = []
    candidates_tok = []
    for c in CANDIDATES:
        # '. ' added to input so that tokenizer understand that first word follows a space.
        tokens = tokenizer.tokenize('. ' + c)[1:]
        candidates.append(tokens)
        candidates_tok.append(tokenizer.convert_tokens_to_ids(tokens))

    for profession, prompt in tqdm(data.items()):
        # original prompt
        example = prompt.replace("{}", profession)
        output_prompt = get_probabilities_for_examples(example, candidates_tok)

        # template prompt
        example = template.replace("{}", profession)
        output_template = get_probabilities_for_examples(example, candidates_tok)

        to_dict[profession] = {
            "prompt cont.": output_prompt,
            "template cont.": output_template
        }

        with open(base + filename + '.json', 'w') as f:
            json.dump(to_dict, f)

    df = pd.DataFrame(to_dict).unstack().reset_index()
    df.columns = ['profession', 'source', 'prob']
    df = pd.concat([df.drop('prob', axis=1), df['prob'].apply(pd.Series)], axis=1)
    df.to_csv(filename + '.csv', index=False)

def construct_data(filename):
    male = ['he', 'his', 'him']
    female = ['she', 'her', 'hers']
    neutral = ['they', 'their', 'them']

    df = pd.read_csv(base + filename + '.csv')
    df = df.set_index(['profession', 'source'])
    
    to_dict = dict()
    cols = df.idxmax(axis=1)

    for i, row in tqdm(df.iterrows()):
        profession = i[0]
        value = row[cols[i]]
        direction = "review me"
        if cols[i] in female:
            value = -value

        if profession in to_dict.keys():
            if abs(to_dict[profession][1]) < abs(value):
                to_dict[profession] = [direction, value, i[1]]

        else:    
            to_dict[profession] = [direction, value, i[1]]

    to_list = [[k, v[0], round(v[1], 1), v[2]] for k, v in to_dict.items()]
    with open(filename + '.json', 'w') as f:
        json.dump(to_list, f)

if __name__ == '__main__':
    filename = 'stereo_prob'
    construct_data(filename)
