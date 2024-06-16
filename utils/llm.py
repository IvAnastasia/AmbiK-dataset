import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, GemmaForCausalLM
import torch

def temperature_scaling(logits, temperature=1):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx
    
class LLM:
    def __init__(self, model_name, generation_settings):
        self.device = "cuda" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mtype = None
        
        if "t5" in model_name or "bart" in model_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.mtype ='d'
        elif 'gemma' in model_name:
            self.model = GemmaForCausalLM.from_pretrained(model_name)
            self.mtype ='ed'
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.mtype ='ed'
            print('gpt2')
            
        
        self.model.to(self.device)  # Move model to CUDA device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_settings = generation_settings

    def filter_logits(self, logits, words, use_softmax=True):
        # Initialize an empty list to collect all token IDs
        token_ids = []
        # Tokenize each word and convert to IDs
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)
    
        token_ids = [t.item() for t in torch.tensor(token_ids, device=self.device)]
        
        count_tokens = dict(Counter(token_ids).most_common())
        token_ids_target = [key for key in count_tokens.keys() if count_tokens[key] == 1 ]
        filtered_logits = [logits[t].item() for t in token_ids_target]
        if use_softmax:
            filtered_logits = temperature_scaling(filtered_logits)
        return dict(zip(words, filtered_logits))


    def generate(self, prompt, return_logits=False):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)  # Move input_ids to CUDA device

        generated_ids = self.model.generate(input_ids, **self.generation_settings)
        generated_text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids][0]
        if return_logits:
            with torch.no_grad():
                if self.mtype=='ed':
                    combined_ids = torch.cat([input_ids, generated_ids[:, 1:]], dim=1)
                    outputs = self.model(combined_ids)
                    logits = outputs.logits[:, input_ids.size(1):]
                else:
                    outputs = self.model(input_ids=input_ids, decoder_input_ids=generated_ids[:, :-1])
                    logits = outputs.logits

                # Create a mask to filter out special tokens
                special_tokens = self.tokenizer.all_special_ids
                mask = torch.ones_like(generated_ids[0, 1:], dtype=torch.bool)
                for token_id in special_tokens:
                    mask &= (generated_ids[0, 1:] != token_id)

                filtered_logits = logits[:, mask] 
            return (generated_text, filtered_logits)
        else:
            return generated_text
            
    def generate_batch(self, prompts, return_logits=False):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
    
        generated_ids = self.model.generate(input_ids, **self.generation_settings)
        generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
    
        if return_logits:
            logits = self._compute_logits(input_ids, generated_ids)
            filtered_logits = self._filter_special_token_logits(generated_ids, logits)
            return (generated_texts, filtered_logits)
        else:
            return generated_texts
    
    def _compute_logits(self, input_ids, generated_ids):
        with torch.no_grad():
            if self.mtype == 'ed':
                combined_ids = torch.cat([input_ids, generated_ids[:, 1:]], dim=1)
                outputs = self.model(combined_ids)
                logits = outputs.logits[:, input_ids.size(1):]
            else:
                outputs = self.model(input_ids=input_ids, decoder_input_ids=generated_ids[:, :-1])
                logits = outputs.logits
        return logits
    
    def _filter_special_token_logits(self, generated_ids, logits):
        special_tokens = self.tokenizer.all_special_ids
        batch_size, seq_length = generated_ids.shape
        mask = torch.ones((batch_size, seq_length - 1), dtype=torch.bool, device=logits.device)
        for token_id in special_tokens:
            mask &= (generated_ids[:, 1:] != token_id)
        filtered_logits = torch.stack([logit[mask[i]] for i, logit in enumerate(logits)])
        return filtered_logits



if __name__ == "__main__":
    model = LLM("google/flan-t5-large", {"max_length": 50, "num_return_sequences": 1})
    print(model.generate("Choose one letter A/B/C?"))
    answer, logits = model.generate("Choose one letter A/B/C?", return_logits=True)
    print(model.filter_logits(logits[0][1], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4']))
