from openai import OpenAI
from dotenv import load_dotenv
from multiprocessing import Pool
import os
import time
import pickle
import json
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], organization="org-6u1yKGMuXAyb3dStdjvmFHMo")
    

def prompt_gpt(prompt: str):
    """
    chatCompletion does not allow batched prompts unfortuantely.
    """
    system_context = """Given two tweets and a target word in the form of "### Tweet1: ... ### Tweet2: ... ### Target Word: ...", 
    Tell me whether the meaning of the target word is different in the two tweets. Furthermore, only tell me yes or no and do not explain."""

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": f"{system_context}"},
        {"role": "user", "content": f"{prompt}"},
    ],
    temperature=0.1,
    n=1,
    frequency_penalty=1.5,
    max_tokens=15)
    # have to sleep to make sure openai limits are not reached
    time.sleep(2)
    return response.choices[0].message.content

def open_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)
    
def load_instances(fn):
    instances = []
    with open(fn) as f:
        for jl_str in f:
            instances.append(json.loads(jl_str))
    
    return instances

def create_prompt(data_entry):
    tweet1 = data_entry["tweet1"]["text"]
    tweet2 = data_entry["tweet2"]["text"]
    word = data_entry["word"]

    return f"### Tweet1: {tweet1} ### Tweet2: {tweet2} ### Target Word: {word}"

def prep_dataset():
    files = ['data/test-codalab-10k.data.jl', 'data/train.data.jl', 'data/trial.data.jl', 'data/validation.data.jl']
    gpt_prompts = []
    for file in files:
        instances = load_instances(file)
        for data in instances:
            gpt_prompts.append(create_prompt(data))
    
    return gpt_prompts

def parallel_prompt_gpt(start, end):
    prompts = prep_dataset()[start:end]

    chatgpt_answers = []
    pbar = tqdm(total=len(prompts))
    for prompt in prompts:
        chatgpt_answers.append(prompt_gpt(prompt))
        pbar.update(1)
    
    return chatgpt_answers

def main():
    num_procs = 4
    prompts = prep_dataset()

    partition = len(prompts) // num_procs

    pooling_partition_arr = []
    for i in range(num_procs):
        if i == num_procs-1:
            pooling_partition_arr.append((i * partition, len(prompts)))
        else:
            pooling_partition_arr.append((i * partition, (i+1) * partition))
    

    print(pooling_partition_arr)
        
    pool = Pool(num_procs)
    results = pool.starmap(parallel_prompt_gpt, pooling_partition_arr)
    pool.close()
    pool.join()
    print(results)
    print(len(results))

    with open('data/gpt3.5_answers.pkl', 'wb') as fp:
        pickle.dump(results, fp)
    
    with open('data/gpt3.5_answers.pkl', 'rb') as fp:
        chatgpt_answers = pickle.load(fp)
        print(chatgpt_answers)
    

if __name__ == "__main__":
    main()
        

    