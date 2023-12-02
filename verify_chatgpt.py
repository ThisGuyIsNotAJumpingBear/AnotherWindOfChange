import pickle 
from utils import load_instances
from collections import defaultdict
from pearson import load_annotator_labels
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

def get_id_to_target_word():
    instances = load_instances()
    
    id_target_dict = {}
    for instance in instances:
        id_target_dict[instance["id"]] = instance["word"]
    
    return id_target_dict

def get_gpt_accuracy(path):
    ids_to_word = get_id_to_target_word()

    with open(path, 'rb') as fp:
        chatgpt_answers = pickle.load(fp)

    word_acc = defaultdict(int)
    total_word_count = defaultdict(int)
    
    for para_lst in chatgpt_answers:
        for response in para_lst:
            target_word = ids_to_word[response["id"]]
            if response["response"] != "1" and response["response"] != "0":
                print("error")
            elif response["response"] == response["label"]:
                word_acc[target_word] += 1
            
            total_word_count[target_word] += 1

    acc_dict = {}
    for key in list(word_acc.keys()):
        acc_dict[key] = word_acc[key] / total_word_count[key]
    
    return acc_dict

def get_gpt_change(path):
    ids_to_word = get_id_to_target_word()

    with open(path, 'rb') as fp:
        chatgpt_answers = pickle.load(fp)

    meaning_change = defaultdict(list)
    
    for para_lst in chatgpt_answers:
        for response in para_lst:
            target_word = ids_to_word[response["id"]]
            if response["response"] != "1" and response["response"] != "0":
                print("error")
            else:
                meaning_change[target_word].append(int(response["response"]))

    mc_dict = {}
    for key in list(meaning_change.keys()):
        mc_dict[key] = np.mean(np.asarray(meaning_change[key]))

    annotator_labels = load_annotator_labels()

    gpt_vec = []
    annotator_vec = []

    for key in annotator_labels.keys():
        gpt_vec.append(float(mc_dict[key]))
        annotator_vec.append(float(annotator_labels[key]))

    
    pearson, p_value = pearsonr(gpt_vec, annotator_vec)
    print(pearson)
    print(p_value)

def get_gpt_f1(path):
    #f1 not a good metric as we have a lot more true negatives

    with open(path, 'rb') as fp:
        chatgpt_answers = pickle.load(fp)

    response_list = []
    label_list = []
    
    for para_lst in chatgpt_answers:
        for response in para_lst:
            label_list.append(response["label"])
            response_list.append(response["response"])
    
    f1 = f1_score(label_list, response_list, average="binary", pos_label="1")
    print(f1)

if __name__ == "__main__":
    print(get_gpt_f1('data/gpt4_answers_qa.pkl'))
    
    

    

    
