import gc
import sys
import glob
import math
from scipy import spatial
from gensim.models import FastText
from gensim.test.utils import common_texts
import numpy as np
import pandas as pd
import spacy
import re

print(glob.glob("*"))
sys.path.append(".")
sys.path.append("./utils")
from llm import LLM
from parse_config import parse_args, parse_config

from lofree.prompts import exmples_generation
nlp = spacy.load("en_core_web_md")

from lofree.pipeline import LoFreeConfig, LoFreePipe


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

    #processed_A = {key: remove_prefix(sentence) for key, sentence in A.items()}
    processed_A = A
    processed_B = [remove_prefix(sentence) for sentence in B]

    def is_similar(sent1, sent2):
        doc1 = nlp(sent1)
        doc2 = nlp(sent2)
        return doc1.similarity(doc2) > threshold
        
    filtered_A = {}
    mapping = {0:'A', 1:'B', 2:'C', 3:'D'}
    processed_A_li =list(processed_A.keys())
    for i in range(len(processed_A)):
        sentence_A = processed_A_li[i]
        if any(is_similar(sentence_A, sentence_B) for sentence_B in processed_B):
            filtered_A[mapping[i]] = processed_A_li[i] 
    print(filtered_A)
    return filtered_A

def calibtation_example():
     target_success = 0.8 
     epsilon = 1-target_success
    
     configs = parse_config("./configs/lofree.yaml" , use_args=True)
     lofree_config = LoFreeConfig(configs)
     lofree = LoFreePipe(lofree_config)

     #Calibration set
     dataset = pd.read_csv("./ambik_dataset/ambik_72.csv")

     calibration_data = []
     for i in range(len(dataset)): #len(dataset)
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
         
        #options, answers_logits = get_logits(lofree, description, task, prefix, action)
        nonconformity_scores, rest_option = get_logits(lofree, description, task, prefix, action)
        right_variants = dataset.loc[i, 'variants_best'].split("\n")
        print('nonconformity_scores')
        print(nonconformity_scores)
        #filtered_options = filter_similar_sentences(options, right_variants)
        filtered_options = filter_similar_sentences(nonconformity_scores, right_variants)
        print(filtered_options)
        success_logits = [nonconformity_scores[value] for key, value in filtered_options.items()]
        calibration_data+=success_logits

     num_calibration_data = len(calibration_data)
     print(calibration_data)
     q_level = np.ceil((num_calibration_data + 1) * (1 - epsilon)) / num_calibration_data
     qhat = np.quantile(calibration_data, q_level)
     return qhat

if __name__ == "__main__":
    print("CP: ", calibtation_example())
