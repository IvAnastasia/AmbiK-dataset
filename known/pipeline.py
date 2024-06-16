import os
import gc
import sys
import glob
import random
import numpy as np
import pandas as pd
import re
import spacy
import re
import tqdm

print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config
from metrics import _calculate_metrics, aggreate, batch_metric_calculation

from knowno.prompts import exmples_generation, question_generation, answer_generation

nlp = spacy.load("en_core_web_md")
CP = 0.5
class KnowNoConfig():
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
    
   # print(examples)
    options = ""
    mapping = {"A": "1", "B":"2", "C": "3", "D": "4"}
    variants = {"A":[], "B":[], "C":[], "D":[]}
    for line in lines:
        for key in variants.keys():    
            if line.startswith(f"{key})") or line.startswith(f"{mapping[key]}.") or line.startswith(f"{mapping[key]})"):
                variants[key].append(line)
   # print(variants)
    for key in variants.keys():
        variants[key] = list(set(variants[key]))
        if len(variants[key]) > 0:
            options+=variants[key][0] +"\n"
            variants[key] = variants[key][0]
        else:
            options += 'do nothing' +"\n"
            variants[key] = 'do nothing'
    return variants, options
            
    
class KnowNoPipe():
    def __init__(self, config=None):
        global CP
        self.config = config
        self.cp = CP #You can use calibrate.py to recalculate 0.9 - llama 0.7 gemma
        self.mapping_1 = ['A', 'B', 'C', 'D']

    def _get_task(self,prompt):
        # Get target task from few shot
        return "We" + "We".join(prompt.split("D")[-1].split("We")[1:])

    def options_prompt(self, escription, task, prefix, action, full_prompt=None):
         if full_prompt is not None:
            prompt = full_prompt
            task = self._get_task(prompt)
         else:
            prompt = exmples_generation.replace('<DESCRIPTION>', description)
            prompt = prompt.replace('<TASK>', task)
            prompt = prompt.replace('<PREFIX>', prefix)
            prompt = prompt.replace('<ACT>', action)
         return prompt

    def format_options(self, prompt, options):
        examples = options.replace(prompt, "")
        options, options_str = format_examples(examples)
        return options, f"{task}\n Options: \n {options_str}"
        
    def predict_examples(self,  description, task, prefix, action, full_prompt=None):
        llm = LLM(self.config.examples_generation['model'],
                  self.config.examples_generation['generation_kwargs'])
        prompt = self.options_prompt(description, task, prefix, action, full_prompt)
        examples = llm.generate(prompt)
        llm = None
        return self.format_options(prompt, options)

    def predict_examples_batch(self, prompts, batch_size=2):
        llm = LLM(self.config.examples_generation['model'],
                  self.config.examples_generation['generation_kwargs'])
        options_full = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size)):  
            options = llm.generate_batch(prompts[i:i+batch_size])
            options_full += options

        formated_options = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            options = options_full[i]
            formated_options.append(self.format_options(prompt, options))
        llm = None
        gc.collect()
        return formated_options
        
    def answer_with_cp(self, tokens_logits):
        possible_options = []
        for key in tokens_logits.keys():
            if tokens_logits[key] > self.cp:
                possible_options.append(key) 
      #  print(possible_options)
        formated_options = []
        for option in possible_options:
            if option.isdigit():  
                option_formated = self.mapping_1[int(option)-1]
            else:
                option_formated = option.upper()
            if option_formated not in formated_options:
                formated_options.append(option_formated)
        return possible_options

    def answer_prompt(self, prompt):
        prompt = prompt.replace("You", "Options")
        prompt = answer_generation.replace('<PROMPT>', prompt)
        return prompt
        
    def generate_answer_batch(self, prompts, batch_size=2):
        llm = LLM(self.config.answering['model'],
                  self.config.answering['generation_kwargs'])
        full_texts = []
        full_logits = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size)):
            texts, logits = llm.generate_batch(prompts[i:i+batch_size], return_logits=True)
            full_texts+=texts
            full_logits+=logits

        filtered_logits_batch = []
        answers = []
        for i in range(len(full_texts)):
         #   print(full_logits[i])
            filtered_logits = llm.filter_logits(full_logits[i][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])
            filtered_logits_batch.append(filtered_logits)
            answer = self.answer_with_cp(filtered_logits)
            answers.append(answer)
        llm = None
        gc.collect()
        return filtered_logits_batch, answers
        
    def generate_answer(self, prompt):
        llm = LLM(self.config.answering['model'],
                  self.config.answering['generation_kwargs'])
        
        prompt = self.answer_prompt(prompt)
        text, logits = llm.generate(prompt, return_logits=True)
        filtered_logits = llm.filter_logits(logits[-1][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])
        llm = None
        return filtered_logits, self.answer_with_cp(filtered_logits)

    def run(self, description, task, prefix, action):
        options, task = self.predict_examples(description, task, prefix, action)
        answers_letter = self.generate_answer(task)[1]
        answers = [options[letter] for letter in answers_letter]
        
        if len(answers)==0:
            return ['cantanswer']
        return answers

    def run_batch(self, prompts):
       options = self.predict_examples_batch(prompts)
       answer_prompts = [self.answer_prompt(option[1]) for option in options]
       logits, answers = self.generate_answer_batch(answer_prompts)
       right_answers = []
    
       for i in range(len(options)):
           option = options[i][0]
           answers_letter = answers[i]
           if len(answers_letter) > 0:
               answers_ = [option[letter] for letter in answers_letter]
           else: 
               answers_ = ['cantanswer']
           right_answers.append(answers_)
       return right_answers
        


def per_task_pipe():
   configs = parse_config("./configs/knowno.yaml" , use_args=True)
   gen_model = configs['examples_generation']['model'].split("/")[1]
   answ_model = configs['answering']['model'].split("/")[1]
   exp_res_dir = f"./{gen_model}_{answ_model}"
   os.makedirs(exp_res_dir, exist_ok=True)

   print()
   print(" Start experiment !", exp_res_dir)
   print()
   knowno_config = KnowNoConfig(configs)
   knowno = KnowNoPipe(knowno_config)

   #Calibration set
   dataset = pd.read_csv("./ambik_dataset/ambik_72_compl.csv")
   
   #Data fo metrics
   amb_type = dataset['ambiguity_type'].values
   keywords = dataset['amb_shortlist'].values
    
   calibration_data = []
   metrics_batch = {'llm_answers':[], 'y_amb_type':[], 'y_amb_keywords':[],
                     'sr':[], 'help_rate': [], 'ambiguity_detection': []}
   for i in range(len(dataset)): 
        description = dataset.loc[i, 'environment_full']
        task = dataset.loc[i, 'ambiguous_task']
        plan = dataset.loc[i, 'plan_for_amb_task'].split('\n')
        point = dataset.loc[i, 'end_of_ambiguity']
        if point == 0:
            prefix = 'Your first action is:'
        else:
            prefix = 'Your previous actions were:\n'
            for act in plan[:point]:
                prefix += act
                prefix += '\n'
        action = plan[point]
        answer = knowno.run(description, task, prefix, action)
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
    

if __name__ == "__main__":

   configs = parse_config("./configs/knowno.yaml" , use_args=True)
   gen_model = configs['examples_generation']['model'].split("/")[1]
   answ_model = configs['answering']['model'].split("/")[1]
   exp_res_dir = f"./{CP}_{gen_model}_{answ_model}"
   os.makedirs(exp_res_dir, exist_ok=True)

   print()
   print(" Start experiment !", exp_res_dir)
   print()
   knowno_config = KnowNoConfig(configs)
   knowno = KnowNoPipe(knowno_config)

   #Calibration set
   dataset = pd.read_csv("./ambik_dataset/ambik_72_test.csv")
   
   #Data fo metrics
   amb_type = dataset['ambiguity_type'].values
   keywords = dataset['amb_shortlist'].values
    
   calibration_data = []
   metrics_batch = {'llm_answers':[], 'y_amb_type':[], 'y_amb_keywords':[],
                     'sr':[], 'help_rate': [], 'ambiguity_detection': []}
   option_prompts = []
   for i in range(len(dataset)): 
        description = dataset.loc[i, 'environment_full']
        task = dataset.loc[i, 'ambiguous_task']
        plan = dataset.loc[i, 'plan_for_amb_task'].split('\n')
        point = dataset.loc[i, 'end_of_ambiguity']
        if point == 0:
            prefix = 'Your first action is:'
        else:
            prefix = 'Your previous actions were:\n'
            for act in plan[:point]:
                prefix += act
                prefix += '\n'
        action = plan[point]
        option_prompt = knowno.options_prompt(description, task, prefix, action)
        option_prompts.append(option_prompt)
       
   right_answers = knowno.run_batch(option_prompts) # NEED TO REMOOVE [:10]
   metrics_batch = batch_metric_calculation(llm_answers_batch=right_answers, y_amb_type_batch=amb_type, y_amb_keywords_batch=keywords)
   agg_metrics = aggreate(metrics_batch)
   agg_metrics = {key:[agg_metrics[key]] for key in agg_metrics}
   agg_metrics_df = pd.DataFrame(agg_metrics)
   agg_metrics_df.to_csv(f"{exp_res_dir}/knowno_agg_metrics_{i}.csv")
            
   metrics = pd.DataFrame(metrics_batch)
   metrics.to_csv(f"{exp_res_dir}/knowno_metrics_{i}.csv")


              
              


        
        
        
    
        
    
