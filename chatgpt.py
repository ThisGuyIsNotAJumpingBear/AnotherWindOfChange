from openai import OpenAI
from dotenv import load_dotenv
from multiprocessing import Pool
import os
import time
import pickle
import json
from tqdm import tqdm
import csv
import numpy as np
from utils import load_instances

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], organization="org-6u1yKGMuXAyb3dStdjvmFHMo")

context_with_date = ''' 
    Given two tweets, their dates and a target word in the form of "### Tweet1: ... Date: ... ### Tweet2: ... Date: ... ### Target Word: ...", 
    Tell me whether the meaning of the target word is different in the two tweets. Furthermore, only respond with 1 if it is a yes and 0 if it is a no. Do not explain.
'''

context = '''
    Given two tweets and a target word in the form of "### Tweet1: ... ### Tweet2: ... ### Target Word: ...", 
    Tell me whether the meaning of the target word is different in the two tweets. Furthermore, only respond with 1 if it is a yes and 0 if it is a no. Do not explain.
''' 
context_qa = '''
You are given two tweets with their respective dates of creation and a question in the format of Tweet-1: ... Tweet-2: ... Question: ...
Answer the question with 1 if it is a yes and 0 if it is a no. Do not explain.
'''

context_qa_amount_change = '''
You are given two tweets with their respective dates of creation and a question in the format of Tweet-1: ... Tweet-2: ... Question: ...
Answer the question with a float number within the range of 0 to 1 where 1 is yes and 0 is no. Do not explain.
'''
    
def prompt_gpt(prompt_obj, context):
    """
    chatCompletion does not allow batched prompts unfortuantely.
    """

    response = client.chat.completions.create(model="gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": f"{context}"},
        {"role": "user", "content": f"{prompt_obj['prompt']}"},
    ],
    temperature=0.1,
    n=1,
    frequency_penalty=1.5,
    max_tokens=15)
    # have to sleep to make sure openai limits are not reached
    time.sleep(2)
    return {"id": prompt_obj["id"], "response": response.choices[0].message.content, "label": prompt_obj["label"]}

def open_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)
    


def create_prompt(data_entry):
    tweet1 = data_entry["tweet1"]["text"]
    tweet2 = data_entry["tweet2"]["text"]
    word = data_entry["word"]

    return f"### Tweet1: {tweet1} ### Tweet2: {tweet2} ### Target Word: {word}"

def create_prompt_with_date(data_entry):
    tweet1 = data_entry["tweet1"]["text"]
    tweet2 = data_entry["tweet2"]["text"]
    word = data_entry["word"]
    date1 = data_entry["tweet1"]["date"]
    date2 =  data_entry['tweet2']["date"]
    return f"### Tweet1: {tweet1} Date: {date1} ### Tweet2: {tweet2} Date: {date2} ### Target Word: {word}"

def create_qa_prompt(data_entry):
    tweet1 = data_entry["tweet1"]["text"]
    tweet2 = data_entry["tweet2"]["text"]
    word = data_entry["word"]
    date1 = data_entry["tweet1"]["date"]
    date2 =  data_entry['tweet2']["date"]

    return f"Tweet-1: {tweet1} Date: {date1} Tweet-2: {tweet2} Date: {date2} Question: Is the meaning of {word} different in the last 2 tweets?"


def load_labels():
    label_files = ['data/test.gold.tsv', 'data/train.labels.tsv', 'data/trial.gold.tsv', 'data/validation.labels.tsv']
    labels = {}
    for file in label_files:
        with open(file) as fd:
            rd = csv.reader(fd, delimiter="\t")
            for row in rd:
                labels[row[0]] = row[1]
    return labels

def prep_dataset(create_prompt_func):
    gpt_prompts = []
    labels = load_labels()
    
    instances = load_instances()
    for data in instances:
        if data["id"] in labels.keys():
            gpt_prompts.append({'id': data["id"], "prompt":create_prompt_func(data), "label": labels[data["id"]]})
    
    return gpt_prompts

def parallel_prompt_gpt(prompts, start, end):
    core_prompts = prompts[start:end]

    chatgpt_answers = []
    pbar = tqdm(total=len(core_prompts))
    for prompt in core_prompts:
        chatgpt_answers.append(prompt_gpt(prompt, context_qa_amount_change))
        pbar.update(1)
    
    return chatgpt_answers

def main():
    num_procs = 4
    prompts = prep_dataset(create_qa_prompt)

    gpt4 = False
    if gpt4:
        idx_arr = np.random.choice(np.arange(len(prompts)), size=200, replace=False)
        scaled_down_prompts = []
    
        for idx in idx_arr:
            scaled_down_prompts.append(prompts[idx])
        
        prompts = scaled_down_prompts

    partition = len(prompts) // num_procs

    pooling_partition_arr = []
    for i in range(num_procs):
        if i == num_procs-1:
            pooling_partition_arr.append((prompts, i * partition, len(prompts)))
        else:
            pooling_partition_arr.append((prompts, i * partition, (i+1) * partition))
    

    print(pooling_partition_arr)
        
    pool = Pool(num_procs)
    results = pool.starmap(parallel_prompt_gpt, pooling_partition_arr)
    pool.close()
    pool.join()
    print(results)
    print(len(results))

    with open('data/gpt3.5_answers_change.pkl', 'wb') as fp:
        pickle.dump(results, fp)
    
    with open('data/gpt3.5_answers_change.pkl', 'rb') as fp:
        chatgpt_answers = pickle.load(fp)
        print(chatgpt_answers)
    

def get_accuracy():
    with open('data/gpt3.5_answers_qa.pkl', 'rb') as fp:
        chatgpt_answers = pickle.load(fp)
    
    count = 0
    total_count = 0
    for para_lst in chatgpt_answers:
        for response in para_lst:
            if response["response"] != "1" and response["response"] != "0":
                print("error")
            elif response["response"] == response["label"]:
                count += 1
            
            total_count += 1
    print(count, total_count)
    return count / total_count
    

if __name__ == "__main__":
    # main()
    with open('data/gpt3.5_answers_change.pkl', 'rb') as fp:
        chatgpt_answers = pickle.load(fp)

    acc = get_accuracy()
    print(acc)
        

    