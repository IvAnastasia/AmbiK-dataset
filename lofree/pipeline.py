import gc
import sys
import glob
import math
from scipy import spatial
from gensim.models import FastText
from gensim.test.utils import common_texts
import numpy as np
import pandas as pd
import random
import os
import spacy

print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config
from metrics import _calculate_metrics, aggreate

from lofree.prompts import exmples_generation

nlp = spacy.load("en_core_web_md")

#генерируем 20 вариантов, считаем для уникальных среди них метрики

class LoFreeConfig():
    def __init__(self, config):
        config_mapping = {
            'examples_generation': ['model', 'generation_kwargs'],
            'answering': ['model', 'generation_kwargs'],
        }
        self.config = config
        self.examples_generation = config['examples_generation']
        self.answering = config['answering']

def format_examples(examples):
    lines = examples.split("\n")
    if "\n" in examples:
        lines = examples.split("\n")
        if len(lines) < 4:
            lines = [x for x in re.split(r'(\d. [\w\s]*.)', examples) if len(x) > 2]         
    else:
        lines = [x for x in re.split(r'(\d. [\w\s]*.)', examples) if len(x) > 2]
    
    print(examples)
    options = ""
    mapping = {"A": "1", "B":"2", "C": "3", "D": "4"}
    variants = {"A":[], "B":[], "C":[], "D":[]}
    for line in lines:
        for key in variants.keys():    
            if line.startswith(f"{key})") or line.startswith(f"{mapping[key]}.") or line.startswith(f"{mapping[key]})"):
                variants[key].append(line)
    print(variants)
    for key in variants.keys():
        variants[key] = list(set(variants[key]))
        if len(variants[key]) > 0:
            options+=variants[key][0] +"\n"
            variants[key] = variants[key][0]
        else:
            options += 'do nothing' +"\n"
            variants[key] = 'do nothing'
    return variants, options

class LoFreePipe():
    def __init__(self, config=None):
        self.config = config
        self.prompting_number = 5
        self.lambda1 = 0.1 #позаменять 
        self.lambda2 = 0.1
        self.lambda_bottom = 0
        self.lambda_up = 2
        self.lambda_step = 0.05
        self.cp = 0.8361353116146786
        #self.cp = 0.001
        self.mapping_1 = ['A', 'B', 'C', 'D']
        self.model_ss = FastText(sentences=common_texts, vector_size=200, window=5, min_count=1, workers=4)
        self.model_ss.save("ft.model")

    def predict_examples(self,  description, task, prefix, action, full_prompt=None):
        llm = LLM(self.config.examples_generation['model'],
                  self.config.examples_generation['generation_kwargs'])

        if full_prompt is not None:
            prompt = full_prompt
            task = self._get_task(prompt)
        else:
            prompt = exmples_generation.replace('<DESCRIPTION>', description)
            prompt = prompt.replace('<TASK>', task)
            prompt = prompt.replace('<PREFIX>', prefix)
            prompt = prompt.replace('<ACT>', action)
        
        examples = llm.generate(prompt)
        llm = None
        return examples

    def answer_with_cp(self, scores):
        possible_options = []
        for key in scores.keys():
            if scores[key] >= self.cp:
                possible_options.append(key) 
        #formated_options = []
        #for option in possible_options:
         #   if option.isdigit():  
          #      option_formated = self.mapping_1[int(option)]
           # else:
            #    option_formated = option.upper()
            #if option_formated not in formated_options:
             #   formated_options.append(option_formated)
        return scores, possible_options
        #return possible_options
   
    def generate_answer(self, prompt): ##OLD
        llm = LLM(self.config.answering['model'],
                  self.config.answering['generation_kwargs'])
        
        prompt = prompt.replace("You", "Options")
        prompt = answer_generation.replace('<PROMPT>', prompt)
        text, logits = llm.generate(prompt, return_logits=True)
        #print(text)
        filtered_logits = llm.filter_logits(logits[-1][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])
        llm = None
        return filtered_logits, self.answer_with_cp(filtered_logits)


    def run(self, description, task, prefix, action):
        #options, task = self.predict_examples(description, task, prefix, action)
        answers_raw, answers = lofree.generate_options(description, task, prefix, action)
        frequences = lofree.frequency(answers)
        normalized_entropy = lofree.normalized_entropy(frequences)
        similarities = lofree.semantic_similarity(frequences)
        nonconformity_scores = lofree.nonconformity_score(frequences, normalized_entropy, similarities)
        nonconformity_scores, rest_options = lofree.answer_with_cp(nonconformity_scores)
        #answers_letter = self.generate_answer(task)[1]
        #answers = [options[letter] for letter in answers_letter]
        
        if len(rest_options)==0:
            return ['cantanswer']
        return rest_options

    
    def generate_options(self,  description, task, prefix, action, full_prompt=None):
        #20 = 5* 4
        llm = LLM(self.config.examples_generation['model'],
                  self.config.examples_generation['generation_kwargs'])

        if full_prompt is not None:
            prompt = full_prompt
            task = self._get_task(prompt)
        else:
            prompt = exmples_generation.replace('<DESCRIPTION>', description)
            prompt = prompt.replace('<TASK>', task)
            prompt = prompt.replace('<PREFIX>', prefix)
            prompt = prompt.replace('<ACT>', action)

        answers_raw = []
        answers = {}
         
        for i in range(self.prompting_number):
            text = llm.generate(prompt)
            text = text.replace(prompt, "")
            options, options_str = format_examples(text)
            for k, v in options.items():
                option = v.lower()
                option = option[3:]
                answers_raw.append(text)
                if option in answers.keys():
                    answers[option] += 1
                else:
                    answers[option] = 1

        llm = None
        self.unique_options = len(answers)
        
        return answers_raw, answers
    
    def frequency(self, answers):
        frequences = {}
        for k, v in answers.items():
            frequences[k] = v/(self.prompting_number * 4)
        return frequences
    
    def normalized_entropy(self, frequences):
        log_sum = 0
        for option, freq in frequences.items():
            log_sum += freq * math.log(freq)
        normalized_entropy = abs(log_sum/ math.log(self.prompting_number))
        return normalized_entropy
    
    def get_sentence_vector(self, sentence):
        words = sentence.split()
        vector = [self.model_ss.wv[word] for word in words if word in self.model_ss.wv]
        if vector:
            return sum(vector) / len(vector)
        else:
            return np.zeros(self.model_ss.vector_size)

    def semantic_similarity(self, frequences):
        similarities = {}
        frequences_sorted = dict(sorted(frequences.items(), key=lambda item: item[1], reverse = True))
        best_option, highest_freq = next(iter(frequences_sorted.items()))
        vector_top =  self.get_sentence_vector(best_option)

        for option, freq in frequences.items():
            vector_cur = self.get_sentence_vector(option)
            if freq != highest_freq:
                similarities[option] = 1 - spatial.distance.cosine(vector_cur, vector_top)
            else:
                similarities[option] = 0

        return similarities

    def nonconformity_score(self, frequences, normalized_entropy, similarities):

        nonconformities = {}

        #self.lambda1
        #self.lambda2 подбираются на стадии валидации

        for option, freq in frequences.items():
            nonconformities[option] = 1 - freq + self.lambda1 * normalized_entropy - self.lambda2 * similarities[option]

        return nonconformities

def get_logits(lofree,  description, task, prefix, action): #def get_logits(knowno, prompt)
    answers_raw, answers = lofree.generate_options(description, task, prefix, action) 
    gc.collect()
    frequences = lofree.frequency(answers)
    #print(frequences)
    normalized_entropy = lofree.normalized_entropy(frequences)
    #print(normalized_entropy)
    similarities = lofree.semantic_similarity(frequences)
    #print(similarities)
    nonconformity_scores = lofree.nonconformity_score(frequences, normalized_entropy, similarities)
    #print(nonconformity_scores)
    
    nonconformity_scores, rest_options = lofree.answer_with_cp(nonconformity_scores)
    #print(nonconformity_scores)
    #print(rest_options)
    #choose = lofree.generate_answer(task)
    
    gc.collect()
    return nonconformity_scores, rest_options[0]

def check_options(task=None, options=None):
    correct_probs = []
    for key in options:
        p = random.random()
        if p>0.5:
            correct_probs.append(options[key])
    return correct_probs
    
def calibration_example():
     target_success = 0.8  #@param {type: "number"} {form-width: "10%"}
     epsilon = 1-target_success
    
     configs = parse_config("./configs/lofree.yaml" , use_args=True)
     lofree_config = LoFreeConfig(configs)
     lofree = LoFreePipe(lofree_config)
     #dataset = "".join(open("./knowno/knowno_dataset/metabot-mc-gen-prompt.txt").readlines())
     dataset = pd.read_csv("./ambik_dataset/ambik_plans.csv") #для примера калибрвоки вообще не нужен
     #full_prompts = dataset.split("--0000--")
     calibration_data = []
     for i in range(3): #len(dataset)
        description = dataset.loc[i, 'environment_full']
        task = dataset.loc[i, 'ambiguous_task']
        plan = dataset.loc[i, 'amb_plan'].split('\n')
        point = dataset.loc[i, 'ambiguity_points']
        if point == 0:
            prefix = 'Your first action is:'
        else:
            prefix = 'Your previous actions were:\n'
            for act in plan[:point]:
                prefix += act
                prefix += '\n'
        action = plan[point]
         
        nonconformity_scores, rest_option = get_logits(lofree, description, task, prefix, action)
        calibration_data+=check_options(options=nonconformity_scores)
     print(calibration_data)
     num_calibration_data = len(calibration_data)
     q_level = np.ceil((num_calibration_data + 1) * (1 - epsilon)) / num_calibration_data
     qhat = np.quantile(calibration_data, q_level)
     return qhat

if __name__ == "__main__":
    configs = parse_config("./configs/lofree.yaml" , use_args=True)
    gen_model = configs['examples_generation']['model'].split("/")[1]
    answ_model = configs['answering']['model'].split("/")[1]
    exp_res_dir = f"./{gen_model}_{answ_model}"
    os.makedirs(exp_res_dir, exist_ok=True)

    print()
    print(" Start experiment !", exp_res_dir)
    print()


    lofree_config = LoFreeConfig(configs)
    lofree = LoFreePipe(lofree_config)

    dataset = pd.read_csv("./ambik_dataset/ambik_72_compl.csv")

   #Data fo metrics
    amb_type = dataset['ambiguity_type'].values
    keywords = dataset['amb_shortlist'].values
    
    calibration_data = []
    metrics_batch = {'llm_answers':[], 'y_amb_type':[], 'y_amb_keywords':[],
                     'sr':[], 'help_rate': [], 'ambiguity_detection': []}


    for i in range(len(dataset)): #len(dataset)
        description = dataset.loc[i, 'environment_full']
        task = dataset.loc[i, 'ambiguous_task']
        plan = dataset.loc[i, 'plan_for_amb_task'].split('\n') #ДЛЯ ТЕСТИРОВАНИЯ ОДНОЗНАЧНЫХ ЗАМЕНИТЬ
        point = dataset.loc[i, 'end_of_ambiguity']
        if point == 0:
            prefix = 'Your first action is:'
        else:
            prefix = 'Your previous actions were:\n'
            for act in plan[:point]:
                prefix += act
                prefix += '\n'
        action = plan[point]
        answer = lofree.run(description, task, prefix, action)
        if isinstance(keywords[i], str):
            sample_keywords = keywords[i].split(",")
        else:
            sample_keywords = -1
        metrics = _calculate_metrics(answer, amb_type[i], sample_keywords)
        for key in metrics:
            metrics_batch[key].append(metrics[key])
        
        if i%10 == 0:
           agg_metrics = aggreate(metrics_batch)
           agg_metrics = {key:[agg_metrics[key]] for key in agg_metrics}
           agg_metrics_df = pd.DataFrame(agg_metrics)
           agg_metrics_df.to_csv(f"{exp_res_dir}/knowno_agg_metrics_{i}.csv")
            
           metrics = pd.DataFrame(metrics_batch)
           metrics.to_csv(f"{exp_res_dir}/knowno_metrics_{i}.csv")
    print(aggreate(metrics_batch))
            
     #description = "Apart from that, in the kitchen there is a bottled water, a bag of fishen chips, and a bag of rice chips."
     #task = "I would like a bag of chips."
     #examples = lofree.predict_examples(description, task)
     #gc.collect()
     #print(examples.split('__task__')[1])
     #answer = examples.split('__task__')[1]
     #choose = lofree.generate_answer(answer)
     #gc.collect()
     #print(choose)
