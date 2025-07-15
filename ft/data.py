import re
from datasets import Dataset, DatasetDict

def remove_special_characters(batch):
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"<>\_\|\…]'
    text = batch["transcript"].lower()
    text = text.replace("\ufeff", "")
    batch["transcript"] = re.sub(chars_to_ignore_regex, '', text).strip() + " "
    return batch


def clean_transcript(example):
    unique_tokens = ['<overlap>','<laugh>','<dtmf>','<foreign>','<breath>','<cough>','<lipsmack>','<ring>','<click>']
    cleaned_text = example["transcript"].lower()
    
    char_mapping = { 'à': 'a', 'ä': 'a', 'ç': 'c', 'é': 'e', 'í': 'i', 'ñ': 'n', 'ó': 'o', 'ô': 'o', 'ù': 'u', 'ú': 'u', 'ü': 'u', 'đ': 'd', 'ı': 'i', 'ạ': 'a', 'ả': 'a', 'ậ': 'a', 'ắ': 'a', 'ế': 'e', 'ệ': 'e', 'ồ': 'o'}
    
    cleaned_text = ''.join(char_mapping.get(char, char) for char in cleaned_text)
    
    for u_token in unique_tokens:
        cleaned_text = cleaned_text.replace(u_token, '') 
    cleaned_text = cleaned_text.strip() 
    
    return {"transcript": cleaned_text}



def extract_all_chars(batch):
    all_text = " ".join(batch["transcript"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}



def prepare_dataset(batch, processor):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    return batch





