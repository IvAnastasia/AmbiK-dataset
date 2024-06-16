import gc
import sys
import glob
import random
import numpy as np
import pandas as pd
import re
import spacy
import re

print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config

from knowno.prompts import exmples_generation, question_generation, answer_generation
nlp = spacy.load("en_core_web_md")

from knowno.pipeline import KnowNoConfig, KnowNoPipe



def get_logits(knowno,  description, task, prefix, action): #def get_logits(knowno, prompt)
     options, task = knowno.predict_examples(description, task, prefix, action) 
     gc.collect()
     choose = knowno.generate_answer(task)
     gc.collect()
     return options, choose[0]

def check_options(task=None, options=None):
    correct_probs = []
    for key in options:
        p = random.random()
        if p>0.5:
            correct_probs.append(options[key])
    return correct_probs


def filter_similar_sentences(A, B, threshold=0.95):
    """
    Фильтрует словарь A, оставляя только те предложения, которые по смыслу похожи на предложения из списка B.

    :param A: Словарь с предложениями для фильтрации
    :param B: Список предложений для сравнения
    :param threshold: Порог схожести для сравнения предложений (по умолчанию 0.7)
    :return: Отфильтрованный словарь
    """
    
    # Функция для удаления частей "A)", "B)", и т.д.
    def remove_prefix(text):
        return re.sub(r'^[A-Z]\)\s*', '', text)

    processed_A = {key: remove_prefix(sentence) for key, sentence in A.items()}
    processed_B = [remove_prefix(sentence) for sentence in B]

    def is_similar(sent1, sent2):
        doc1 = nlp(sent1)
        doc2 = nlp(sent2)
        return doc1.similarity(doc2) > threshold
        
    filtered_A = {}
    for key, sentence_A in processed_A.items():
        if any(is_similar(sentence_A, sentence_B) for sentence_B in processed_B):
            filtered_A[key] = A[key] 
    return filtered_A

def calibtation_example():
     target_success = 0.8 
     epsilon = 1-target_success
    
     configs = parse_config("./configs/knowno.yaml" , use_args=True)
     knowno_config = KnowNoConfig(configs)
     knowno = KnowNoPipe(knowno_config)

     #Calibration set
     dataset = pd.read_csv("./ambik_dataset/ambik_72.csv")

     calibration_data = []
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
         
        options, answers_logits = get_logits(knowno, description, task, prefix, action)
        right_variants = dataset.loc[i, 'variants_best'].split("\n")
        filtered_options = filter_similar_sentences(options, right_variants)
        success_logits = [answers_logits[key] for key in filtered_options]
        calibration_data+=success_logits

     num_calibration_data = len(calibration_data)
     print(calibration_data)
     q_level = np.ceil((num_calibration_data + 1) * (1 - epsilon)) / num_calibration_data
     qhat = np.quantile(calibration_data, q_level)
     return qhat

if __name__ == "__main__":
    print("CP: ", calibtation_example())
    
    