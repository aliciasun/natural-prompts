import json
import csv

def parse_saliency(file_name):
    f = open(file_name, 'r')
    db = json.load(f)
    return db

def get_variance(file_name, out_file):
    data_path = 'professions_prompts.json'
    with open(data_path) as f:
        data = json.load(f)
    professions = list(data.keys())

    f = open(file_name, 'r')
    saliency = json.load(f)

    header = ['profession', 'length', 'means', 'variances']
    csv_data = []

    i = 0
    means = []
    variances = []
    for prompt in saliency:
        
        index = i//5
        # saliency distribution
        results = prompt["attributions"][0]

        # calculate mean
        m = sum(results) / len(results)
        means.append(m)

        # calculate variance using a list comprehension
        var_res = sum((xi - m) ** 2 for xi in results) / len(results)
        variances.append(var_res)

        if i % 5 == 4:
            row_data = []
            row_data.append(professions[index])
            row_data.append(len(results))
            row_data.append(means)
            row_data.append(variances)
            csv_data.append(row_data)
            means = []
            variances = []

        i += 1

    with open(out_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)


def attention():
    data_path = 'professions_prompts.json'
    with open(data_path) as f:
        data = json.load(f)
    professions = list(data.keys())

    f = open("saliency_scores/professions_prompts_saliency.json", 'r')
    saliency = json.load(f)

    # for p,v in data.items():
    #     print("WHEE", k, v)
    #     bias_text = v.replace('{}',k)


    header = ['profession', 'first', 'score', 'second', 'score', 'difference']
    csv_data = []

    i = 0
    for prompt in saliency:  
        index = i//5

        # saliency distribution
        scores = prompt["attributions"][0]
        tokens = prompt["tokens"]

        sentence_prompt = data[professions[index]]
        prompt_words = sentence_prompt.split()
        profession_index = prompt_words.index('{}')
        # if profession_index != 1:
        #     print(professions[index])
        sentence = sentence_prompt.replace('{}', professions[index])

        # n = 2
        # result = [scores.index(i) for i in sorted(scores, reverse=True)][:n]

        # row_data = []
        # row_data.append(professions[index])
        # row_data.append(scores[result[0]])
        # row_data.append(scores[result[1]])
        # row_data.append(scores[result[0]]-scores[result[1]])
        # csv_data.append(row_data)

        i += 1

def template_attention():
    data_path = 'professions_prompts.json'
    with open(data_path) as f:
        data = json.load(f)
    professions = list(data.keys())

    f = open("saliency_scores/professions_template1_saliency.json", 'r')
    saliency = json.load(f)

    header = ['profession', 'token indices', 'first', 'score', 'second', 'score', 'difference', 'first in profession', 'second in profession']
    csv_data = []

    i = 0
    for prompt in saliency:  
        index = i//5

        # saliency distribution
        scores = prompt["attributions"][0]
        tokens = prompt["tokens"]

        sentence_prompt = "The {} said that"
        prompt_words = sentence_prompt.split()

        profession_token_indices = []
        profession_scores = []
        for j in range(1, len(tokens)-3):
            profession_token_indices.append(j)
            profession_scores.append(scores[j])

        n = 2
        result = [scores.index(i) for i in sorted(scores, reverse=True)][:n]

        
        sum_profession = sum(profession_scores)
        avg_profession = sum_profession/len(profession_scores)
   
        row_data = []
        row_data.append(professions[index])
        row_data.append(profession_token_indices)
        # row_data.append(sum_profession)
        # row_data.append(avg_profession)
        row_data.append(tokens[result[0]]['token'])
        row_data.append(scores[result[0]])
        row_data.append(tokens[result[1]]['token'])
        row_data.append(scores[result[1]])
        row_data.append(scores[result[0]]-scores[result[1]])
        first_in_profession = "Y!" if result[0] in profession_token_indices else "N!"
        row_data.append(first_in_profession)
        second_in_profession = "y!" if result[1] in profession_token_indices else "n!"
        row_data.append(second_in_profession)
        csv_data.append(row_data)
        i += 1


    with open('template_attention_individual_token.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)

def prompt_attention():
    data_path = 'professions_prompts.json'
    with open(data_path) as f:
        data = json.load(f)
    professions = list(data.keys())

    f = open("saliency_scores/professions_prompts_saliency.json", 'r')
    saliency = json.load(f)

    header = ['profession', 'token indices', 'first', 'score', 'second', 'score', 'difference', 'first in profession', 'second in profession']
    csv_data = []

    i = 0
    for prompt in saliency:  
        index = i//5

        # saliency distribution
        scores = prompt["attributions"][0]
        tokens = prompt["tokens"]

        sentence_prompt = data[professions[index]]
        prompt_words = sentence_prompt.split()

        profession_token_indices = []
        profession_scores = []
        profession = professions[index]

        ind = prompt_words.index('{}')

        while tokens[ind]['token'] in professions[index]:
            profession_token_indices.append(ind)
            profession_scores.append(scores[ind])
            ind += 1

        n = 2
        result = [scores.index(i) for i in sorted(scores, reverse=True)][:n]

        
        # sum_profession = sum(profession_scores)
        # avg_profession = sum_profession/len(profession_scores)
   
        row_data = []
        row_data.append(professions[index])
        row_data.append(profession_token_indices)
        # row_data.append(sum_profession)
        # row_data.append(avg_profession)
        row_data.append(tokens[result[0]]['token'])
        row_data.append(scores[result[0]])
        row_data.append(tokens[result[1]]['token'])
        row_data.append(scores[result[1]])
        row_data.append(scores[result[0]]-scores[result[1]])
        first_in_profession = "Y!" if tokens[result[0]]['token'] in profession else "N!"
        row_data.append(first_in_profession)
        second_in_profession = "y!" if tokens[result[1]]['token'] in profession else "n!"
        row_data.append(second_in_profession)
        csv_data.append(row_data)
        i += 1


    with open('prompts_attention_individual_token.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)

def template_aggregate_attention():
    data_path = 'professions_prompts.json'
    with open(data_path) as f:
        data = json.load(f)
    professions = list(data.keys())

    f = open("saliency_scores/professions_template1_saliency.json", 'r')
    saliency = json.load(f)

    header = ['profession', 'token indices', 'profession tokens', 'score sum', 'last token', 'score', 'difference']
    csv_data = []

    i = 0
    for prompt in saliency:  
        index = i//5

        # saliency distribution
        scores = prompt["attributions"][0]
        tokens = prompt["tokens"]

        sentence_prompt = "The {} said that"
        prompt_words = sentence_prompt.split()

        profession_token_indices = []
        profession_tokens = []
        profession_scores = []
        for j in range(1, len(tokens)-3):
            profession_token_indices.append(j)
            profession_tokens.append(tokens[j]['token'])
            profession_scores.append(scores[j])
        
        sum_profession = sum(profession_scores)
        avg_profession = sum_profession/len(profession_scores)
   
        row_data = []
        row_data.append(professions[index])
        row_data.append(profession_token_indices)
        row_data.append(profession_tokens)
        row_data.append(sum_profession)
        row_data.append(tokens[len(scores)-1]['token'])
        row_data.append(scores[len(scores)-1])
        row_data.append(sum_profession - scores[len(scores)-1])
        csv_data.append(row_data)
        
        i += 1


    with open('template_attention_sum_token.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)

def prompt_aggregate_attention():
    data_path = 'professions_prompts.json'
    with open(data_path) as f:
        data = json.load(f)
    professions = list(data.keys())

    f = open("saliency_scores/professions_prompts_saliency.json", 'r')
    saliency = json.load(f)

    header = ['profession', 'token indices', 'profession tokens', 'score sum', 'last token', 'score', 'difference']
    csv_data = []

    i = 0
    for prompt in saliency:  
        index = i//5

        # saliency distribution
        scores = prompt["attributions"][0]
        tokens = prompt["tokens"]

        sentence_prompt = data[professions[index]]
        prompt_words = sentence_prompt.split()

        profession_token_indices = []
        profession_tokens = []
        profession_scores = []


        profession = professions[index]
        pointer = 0
        start = 0
        end = -1
        # find profession tokens
        
        for j in range(len(tokens)):
            token = tokens[j]['token'].strip()
            if token in profession[pointer:] and profession[pointer:].index(token) == 0:
                pointer += len(token)
                if pointer == len(profession):
                    end = j
                    break
                while profession[pointer] == ' ':
                    pointer += 1
            else:
                pointer = 0
                if token in profession[pointer:] and profession[pointer:].index(token) == 0:
                    start = j
                    pointer += len(token)
                else:
                    start = j + 1

        for j in range(start, end+1):
            profession_token_indices.append(j)
            profession_tokens.append(tokens[j]['token'])
            profession_scores.append(scores[j])
        

        sum_profession = sum(profession_scores)
   
        row_data = []
        row_data.append(profession)
        row_data.append(profession_token_indices)
        row_data.append(profession_tokens)
        row_data.append(sum_profession)
        row_data.append(tokens[len(scores)-1]['token'])
        row_data.append(scores[len(scores)-1])
        row_data.append(sum_profession - scores[len(scores)-1])
        csv_data.append(row_data)
        
        i += 1


    with open('prompt_attention_sum_token.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_data)


if __name__ == '__main__':
    # variance
    # get_variance("professions_template1_saliency.json", "professions_template1_variance.csv")

    # profession - how many times is that the highest
    prompt_aggregate_attention()
    # template_attention()

    # sum up profession, then take average of saliency scores over all of the prompts ! ! ! ! whoop



