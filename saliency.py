import os
import json

import ecco
from transformers import GPT2Tokenizer

#load dataset
data_path = 'data/bold/annotate/professions_prompts.json'
with open(data_path) as f:
  data = json.load(f)

lm = ecco.from_pretrained('distilgpt2', activations=True)

lm.tokenizer('metalsmith is a person fashioning useful')

len(data.items())

template = "The {} said that"
template2 = "The {} ate because"
bias_text = template2.replace('{}', 'metalsmith')
output1 = lm.generate(bias_text, generate=1, do_sample=True)
saliency = output1.saliency(printJson=True)
writefile.write(json.dumps(saliency, indent=4, sort_keys=False))
writefile.write(",\n")

with open('/content/nlp-bias/xlarge_professions_templates_saliency.txt', 'w') as writefile:
  writefile.write("[")

template = "The {} said that"
with open('/content/nlp-bias/xlarge_professions_templates_saliency.txt', 'a') as writefile:
  for k,v in data.items():
    bias_text = template.replace('{}',k)
    output1 = lm.generate(bias_text, generate=1, do_sample=True)
    saliency = output1.saliency(printJson=True)
    writefile.write(json.dumps(saliency, indent=4, sort_keys=False))
    writefile.write(",\n")