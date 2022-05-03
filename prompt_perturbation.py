
import argparse
import json
import os
import torch
import logging
import pandas as pd
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# from model import Model

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

RANDOM_TEMPLATE = [
    '<|endoftext|>',
    'A',
    'An',
    'The',
    'This',
    'That',
    'And',
    'But',
    'Yet',
    'If',
    'Although',
    'So',
    'Because',
    'Therefore',
    'Thus',
    'Since',
    'While',
    'Where',
    'When',
    'Whereas',
    'While',
    'In Fact',
    'As a result',
    'Indeed',
    'For example'
]

# template = TEMPLATE[0]
CANDIDATES = ["he", "she", "they", "his", "her", "theirs", "him", "hers", "them"]

def condidates_to_tokens(tokenizer):
    candidates = []
    candidates_tok = []
    for c in CANDIDATES:
        # '. ' added to input so that tokenizer understand that first word follows a space.
        tokens = tokenizer.tokenize('. ' + c)[1:]
        candidates.append(tokens)
        candidates_tok.append(tokenizer.convert_tokens_to_ids(tokens))
    return candidates_tok



def get_probabilities_for_examples(context, candidates, model, attention_mask = None, return_all_prob=False):
    outputs = [c[0] for c in candidates]
    if attention_mask is not None:
        logits, past = model(context, attention_mask=attention_mask)[:2]
    else:
        logits, past = model(context)[:2]
   
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    if return_all_prob:
        return probs
    # return probs
    else:
        return {k: v for k, v in zip(CANDIDATES, probs[:, outputs].tolist()[0])}


def uncertainty_measure(probs):
    return torch.var_mean(probs, dim=1)

def summary_statistics(data):
    length = []
    length_t = []
    word_length = []
    word_length_t = []

    for profession, prompt in tqdm(data.items()):
        prompt.replace("{}","")
        length.append(len(prompt.split()))
        word_length.extend([len(x) for x in prompt.split()])

    print(np.std(length))
    print(np.std(word_length))
    for template in TEMPLATE:
        template.replace("{}","")
        length_t.append(len(template.split()))
        word_length_t.extend([len(x) for x in template.split()])

    print(np.std(length_t))
    print(np.std(word_length_t))






def prompt_gpt2(data, model, tokenizer):
    to_dict = defaultdict(list)
    candidates_tok = condidates_to_tokens(tokenizer)
    
    for profession, prompt in tqdm(data.items()):
        # original prompt
        example = prompt.replace("{}", profession)
        output_prompt = get_probabilities_for_examples(example, candidates_tok, model)
        var_p, mean_p = uncertainty_measure(output_prompt)
        # max_p = torch.max(output_prompt,dim=1)[0].detach().numpy()[0]
        v = torch.topk(output_prompt, 2, dim=1)[0][0]
        top_one, top_two = v[0].detach().numpy(), v[1].detach().numpy()
        to_dict['profession'].append(profession)
        to_dict['prompt_max'].append(top_one)
        to_dict['prompt_gap'].append(top_one-top_two)
        to_dict['prompt_mean'].append(mean_p.detach().numpy()[0])
        to_dict['prompt_var'].append(var_p.detach().numpy()[0])


        # template prompt
        prob_top_one = []
        prob_top_two = []
        means_t = []
        vars_t = []
        for template in TEMPLATE:
            example = template.replace("{}", profession)
            output_template = get_probabilities_for_examples(example, candidates_tok, model)
            var_t, mean_t = uncertainty_measure(output_template)
            v = torch.topk(output_template, 2, dim=1)[0][0]
            top_one, top_two = v[0].detach().numpy(), v[1].detach().numpy()
            prob_top_one.append(top_one)
            prob_top_two.append(top_two)
            means_t.append(mean_t.detach().numpy()[0])
            vars_t.append(var_t.detach().numpy()[0])
        

        # to_dict['prompt mean'].append(mean_p.detach().numpy())
        # to_dict['prompt var'].append(var_p.detach().numpy())
        to_dict['template_mean'].append(sum(means_t)/len(means_t))
        to_dict['template_var'].append(sum(vars_t)/len(vars_t))
        to_dict['template_max'].append(sum(prob_top_one)/len(prob_top_two))
        to_dict['template_gap'].append((sum(prob_top_one)-sum(prob_top_two))/len(prob_top_two))
        # to_dict['kl_with_first'] = F.kl_div(output_prompt, reduction="mean")

    df = pd.DataFrame.from_dict(to_dict)
    return df

def random_prompt(model, tokenizer):
    candidates_tok = condidates_to_tokens(tokenizer)
    outputs = []
    for example in RANDOM_TEMPLATE:
        output = {}
        output['context'] = example
        tokens = tokenizer.encode(example, add_special_tokens=False, return_tensors="pt")
        output.update(get_probabilities_for_examples(tokens, candidates_tok, model))
        outputs.append(output)
    df = pd.DataFrame(outputs)
    return df


def masked_prompt(data, model, tokenizer, gpt2_version):
    contexts = ["", "person", "man", "woman"]
    candidates_tok = condidates_to_tokens(tokenizer)
    print(gpt2_version)
    for template in TEMPLATE:
        template_name = template.replace(" ","_").replace("{}","")
        output_dir = 'data/masked/{0}'.format(template_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        file_path = 'mask_prob_{0}.csv'.format(gpt2_version)
        output_path = os.path.join(output_dir, file_path)
        to_dict = defaultdict(dict)
        for profession, prompt in tqdm(data.items()):
            example = template.replace("{}", profession)
            context = tokenizer.encode(example, add_special_tokens=False, return_tensors="pt")
            profession_id = tokenizer.encode('. ' + profession)[1:]
            attention_mask = torch.ones_like(context[0])
            attention_mask[1:1+len(profession_id)]=0
            output_orig = get_probabilities_for_examples(context, candidates_tok, model)
            output_masked = get_probabilities_for_examples(context, candidates_tok, model, attention_mask=attention_mask)
            to_dict[profession]["template_original"] = output_orig
            to_dict[profession]["template_mask"] = output_masked
            # original prompt
            for c in contexts:
                example = template.replace("{}", c)
                context = tokenizer.encode(example, add_special_tokens=False, return_tensors="pt")
                output = get_probabilities_for_examples(context, candidates_tok, model)
                to_dict[profession]["template_"+c] = output
        
        df = pd.DataFrame(to_dict).unstack().reset_index()
        df.columns = ['profession', 'source', 'prob']
        df = pd.concat([df.drop('prob', axis=1), df['prob'].apply(pd.Series)], axis=1)
        df.to_csv(output_path, index=False)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='name of dataset')
    parser.add_argument('--gpt2', type=str, default='distilgpt2',
                        help='gpt2 version')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device')
    parser.add_argument('--test', type=str, default='masked',
                        help='device')
   
    args = parser.parse_args()
    gpt2_version = args.gpt2
    device = args.device
    dataset = args.dataset
    gpt2_versions = ['distilgpt2','gpt2','gpt2-medium','gpt2-large']


    MAX_LENGTH = 10
    SAMPLES = 5
    test = args.test


    base = 'data/profession/'
    input_path = base + 'professions_prompts.json'
    with open(input_path, 'rb') as f:
        data = json.load(f)
    # gpt2_version = ['distilgpt2']

    for gpt2 in gpt2_versions:
        model_path='models/{0}'.format(gpt2)
        if os.path.exists(model_path):
            tokenizer = GPT2Tokenizer.from_pretrained(model_path,local_files_only=True)
            model = GPT2LMHeadModel.from_pretrained(model_path,local_files_only=True, pad_token_id=tokenizer.eos_token_id, cache_dir=model_path)
            print(len(model.transformer.h))
            print(model.config.n_head)

        else:
            tokenizer = GPT2Tokenizer.from_pretrained(gpt2)
            model = GPT2LMHeadModel.from_pretrained(gpt2)
        if test == 'uncertainty':
            output_path = base+'uncertainty_results_{0}.csv'.format(gpt2)
            df = prompt_gpt2(data, model, tokenizer)
            df.to_csv(output_path, index=False)
        elif test == 'masked':
            masked_prompt(data,model,tokenizer,gpt2)
            # df.to_csv(output_path, index=False)
        elif test == 'summary':
            summary_statistics(data)
        else:
            output_path = 'random_prompt/random_prompt_{0}.csv'.format(gpt2)
            df = random_prompt(model, tokenizer)
            df.to_csv(output_path, index=False)





    # model = Model(output_attentions=True, gpt2_version=gpt2_version, device=device)

