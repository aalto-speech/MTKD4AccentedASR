import os
import gc
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, 
    Wav2Vec2Config, Wav2Vec2ForCTC, AdamW, get_scheduler
)
from datasets import Dataset, DatasetDict, Audio, load_from_disk, concatenate_datasets
import evaluate

from data import remove_special_characters, clean_transcript, prepare_dataset, merge_datasets
from logits import add_avg_logits, load_dataset_logits, compute_avg_logits, merge_logits, merge_multiple_logits_datasets
from train import train, train_mtkd, train_mtkd_v2
from devel import devel
from test import test
from loss import LossWithLearnableWeights
from collator import DataCollatorCTCWithPadding

import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--TRAIN_ACCENT", help="Accent on which the model will be trained", type=str, default="American", 
        choices=["American", "Canadian", "Indian", "Korean", "Russian", "British", "Chinese", "Japanese", "Portuguese", "Spanish", "AESRC200H", "EDACCdevel", "AESRC200H+EDACCdevel"]
    )
    parser.add_argument(
        "--DEVEL_ACCENT", help="Accent on which the model performance will be evaluated during training", type=str, default="Canadian", 
        choices=["American", "Canadian", "Indian", "Korean", "Russian", "British", "Chinese", "Japanese", "Portuguese", "Spanish"]
    )
    parser.add_argument(
        "--TEST_ACCENT", help="Accent on which the model performance will be tested", type=str, default="Indian", 
        choices=["American", "Canadian", "Indian", "Korean", "Russian", "British", "Chinese", "Japanese", "Portuguese", "Spanish", "AESRC20H", "AESRC10Accents", "AESRC10Accents+AESRC20H", "EDACCtest"]
    )
    parser.add_argument("--N_EPOCHS", help="Number of Epochs", type=int, default=20)
    parser.add_argument("--BATCH_SIZE", help="Batch Size", type=int, default=16)
    parser.add_argument("--LEARNING_RATE", help="Learning Rate", type=float, default=1e-4)
    parser.add_argument("--TRAINING", help="Indicates whether the model will be trained (1) or not (0)", type=int, default=1, choices=[0, 1])
    
    args = parser.parse_args()
    
    # ------------------------------------------------
    
    train_accent = str(args.TRAIN_ACCENT)
    devel_accent = str(args.DEVEL_ACCENT)
    test_accent = str(args.TEST_ACCENT)
    batch_size = int(args.BATCH_SIZE) 
    num_epochs = int(args.N_EPOCHS) 
    num_epochs = 20
    training = int(args.TRAINING)
    learning_rate = float(args.LEARNING_RATE) # 1e-4
    learning_rate = 1e-4 # 5e-5
    vocab_size = 32 # default
    start_epoch = 0 if training else num_epochs
    start_epoch = 0
    seed = 42
    model_checkpoint_name = "facebook/wav2vec2-base-960h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------------------------------------
    
    if train_accent == "American":
        dataset_train = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/American English Speech Data")
    elif train_accent == "Canadian":
        dataset_train = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Canadian English Speech Data")
    elif train_accent == "Indian":
        dataset_train = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Indian English Speech Data")
    elif train_accent == "Korean":
        dataset_train = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Korean Speaking English Speech Data")
    elif train_accent == "Russian":
        dataset_train = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Russian Speaking English Speech Data")
    elif train_accent == "British":
        dataset_train = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/British English Speech Data")
    elif train_accent == "Chinese":
        dataset_train = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Chinese Speaking English Speech Data")
    elif train_accent == "Japanese":
        dataset_train = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Japanese Speaking English Speech Data")
    elif train_accent == "Portuguese":
        dataset_train = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Portuguese Speaking English Speech Data")
    elif train_accent == "Spanish":
        dataset_train = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Spanish Speaking English Speech Data")
    elif train_accent == "AESRC200H":
        all_accents = [
            'American English Speech Data',
            'Canadian English Speech Data',       
            'Indian English Speech Data',             
            'Korean Speaking English Speech Data',      
            'Russian Speaking English Speech Data', 
            'British English Speech Data',   
            'Chinese Speaking English Speech Data',  
            'Japanese Speaking English Speech Data',  
            'Portuguese Speaking English Speech Data',  
            'Spanish Speaking English Speech Data',
        ]
        datasets_list = []
        for accent in sorted(set(all_accents)):
            accented_dataset = load_from_disk(f"/scratch/elec/puhe/c/AESRC/data_hf_format/train/{accent}")
            datasets_list.append(accented_dataset)
            del accented_dataset
        dataset_train = concatenate_datasets(datasets_list)
        dataset_train = dataset_train.shuffle(seed=seed)
        #
    elif train_accent == "AESRC200H+EDACCdevel":
        all_accents = [
            'American English Speech Data',
            'Canadian English Speech Data',       
            'Indian English Speech Data',             
            'Korean Speaking English Speech Data',      
            'Russian Speaking English Speech Data', 
            'British English Speech Data',   
            'Chinese Speaking English Speech Data',  
            'Japanese Speaking English Speech Data',  
            'Portuguese Speaking English Speech Data',  
            'Spanish Speaking English Speech Data',
        ]
        datasets_list = []
        for accent in sorted(set(all_accents)):
            accented_dataset = load_from_disk(f"/scratch/elec/puhe/c/AESRC/data_hf_format/train/{accent}")
            accented_dataset = accented_dataset.cast_column("audio", Audio(sampling_rate=16000))
            datasets_list.append(accented_dataset)
            del accented_dataset
        aesrc200h = concatenate_datasets(datasets_list)
        
        edacc = load_from_disk("/scratch/elec/puhe/c/EDACC/data_hf_format/edacc")
        edacc = DatasetDict({
            split: dataset.rename_column("text", "transcript")
            for split, dataset in edacc.items()
        })
        edacc = edacc.map(clean_transcript)
        edacc_devel = edacc['validation']
        del edacc
        edacc_devel = edacc_devel.remove_columns(
            ["speaker", "raw_accent", "gender", "l1"]
        )
        edacc_devel = edacc_devel.cast_column("audio", Audio(sampling_rate=16000))
        edacc_devel = edacc_devel.map(lambda x: {"audio_path": "A dummy audio path"})
        
        dataset_train = concatenate_datasets([aesrc200h, edacc_devel])
        dataset_train = dataset_train.shuffle(seed=seed)
        #
        
    elif train_accent == "EDACCdevel":
        edacc = load_from_disk("/scratch/elec/puhe/c/EDACC/data_hf_format/edacc")
        edacc = DatasetDict({
            split: dataset.rename_column("text", "transcript")
            for split, dataset in edacc.items()
        })
        edacc = edacc.map(clean_transcript)
    
        dataset_train = edacc['validation']
        del edacc
        dataset_train = dataset_train.remove_columns(
            ["speaker", "raw_accent", "gender", "l1"]
        )
        dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16000))
        dataset_train = dataset_train.map(lambda x: {"audio_path": "A dummy audio path"})
    
        dataset_train = dataset_train.shuffle(seed=seed)
        # 
    else:
        raise ValueError("Training Dataset Mismatch")
    
    
    # ------------------------------------------------
    
    
    if devel_accent == "American":
        dataset_devel = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/American English Speech Data")
    elif devel_accent == "Canadian":
        dataset_devel = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Canadian English Speech Data")
    elif devel_accent == "Indian":
        dataset_devel = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Indian English Speech Data")
    elif devel_accent == "Korean":
        dataset_devel = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Korean Speaking English Speech Data")
    elif devel_accent == "Russian":
        dataset_devel = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Russian Speaking English Speech Data")
    elif devel_accent == "British":
        dataset_devel = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/British English Speech Data")
    elif devel_accent == "Chinese":
        dataset_devel = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Chinese Speaking English Speech Data")
    elif devel_accent == "Japanese":
        dataset_devel = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Japanese Speaking English Speech Data")
    elif devel_accent == "Portuguese":
        dataset_devel = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Portuguese Speaking English Speech Data")
    elif devel_accent == "Spanish":
        dataset_devel = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Spanish Speaking English Speech Data")
    else:
        raise ValueError("Devel Dataset Mismatch")
    
    # ------------------------------------------------
    
    if test_accent == "American":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/American English Speech Data")
    elif test_accent == "Canadian":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Canadian English Speech Data")
    elif test_accent == "Indian":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Indian English Speech Data")
    elif test_accent == "Korean":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Korean Speaking English Speech Data")
    elif test_accent == "Russian":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Russian Speaking English Speech Data")
    elif test_accent == "British":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/British English Speech Data")
    elif test_accent == "Chinese":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Chinese Speaking English Speech Data")
    elif test_accent == "Japanese":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Japanese Speaking English Speech Data")
    elif test_accent == "Portuguese":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Portuguese Speaking English Speech Data")
    elif test_accent == "Spanish":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/train/Spanish Speaking English Speech Data")
    elif test_accent == "AESRC20H":
        dataset_test = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/test")
        dataset_test = dataset_test.map(lambda x: {'accent': 'AESRC20H'})
    elif test_accent == "AESRC10Accents":
        all_accents_aesrc = [
            'American English Speech Data',
            'Canadian English Speech Data',          
            'Indian English Speech Data',             
            'Korean Speaking English Speech Data',      
            'Russian Speaking English Speech Data', 
            'British English Speech Data',   
            'Chinese Speaking English Speech Data',  
            'Japanese Speaking English Speech Data',  
            'Portuguese Speaking English Speech Data',  
            'Spanish Speaking English Speech Data', 
        ]
        datasets_list = []
        for accent in sorted(set(all_accents_aesrc)):
            accented_dataset = load_from_disk(f"/scratch/elec/puhe/c/AESRC/data_hf_format/train/{accent}")
            datasets_list.append(accented_dataset)
            del accented_dataset
        dataset_test = concatenate_datasets(datasets_list)
        # 
    elif test_accent == "AESRC10Accents+AESRC20H":
        all_accents_aesrc = [
            'American English Speech Data',
            'Canadian English Speech Data',          
            'Indian English Speech Data',             
            'Korean Speaking English Speech Data',      
            'Russian Speaking English Speech Data', 
            'British English Speech Data',   
            'Chinese Speaking English Speech Data',  
            'Japanese Speaking English Speech Data',  
            'Portuguese Speaking English Speech Data',  
            'Spanish Speaking English Speech Data', 
        ]
        datasets_list = []
        for accent in sorted(set(all_accents_aesrc)):
            accented_dataset = load_from_disk(f"/scratch/elec/puhe/c/AESRC/data_hf_format/train/{accent}")
            datasets_list.append(accented_dataset)
            del accented_dataset
        aesrc20h = load_from_disk("/scratch/elec/puhe/c/AESRC/data_hf_format/test")
        aesrc20h = aesrc20h.map(lambda x: {'accent': 'AESRC20H'})
        datasets_list.append(aesrc20h)
        del aesrc20h
        dataset_test = concatenate_datasets(datasets_list)
        #
    elif test_accent == "EDACCtest":
        edacc = load_from_disk("/scratch/elec/puhe/c/EDACC/data_hf_format/edacc")
        edacc = DatasetDict({
            split: dataset.rename_column("text", "transcript")
            for split, dataset in edacc.items()
        })
        edacc = edacc.map(clean_transcript)
        
        edacc = edacc.filter(
            lambda example: example['transcript'] != "ignore_time_segment_in_scoring",
            batched=False
        )
    
        dataset_test = edacc['test']
        del edacc
        dataset_test = dataset_test.remove_columns(
            ["speaker", "raw_accent", "gender", "l1"]
        )
        dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=16000))
        dataset_test = dataset_test.map(lambda x: {"audio_path": "A dummy audio path"})
    
        # dataset_test = dataset_test.shuffle(seed=seed)
        # 
    else:
        raise ValueError("Test Dataset Mismatch")
    
    # ------------------------------------------------
    
    if training:
        dataset = DatasetDict({
            'train': dataset_train,
            'devel': dataset_devel,
            'test': dataset_test.select(range(500)),
        })
    else:
        dataset = DatasetDict({
            'train': dataset_train.select(range(500)),
            'devel': dataset_devel.select(range(500)),
            'test': dataset_test,
        })
    
    dataset = dataset.map(remove_special_characters)
    print(dataset)
    
    vocab_dict = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}

    vocab_dir = os.path.join("/m/triton/scratch/elec/t405-puhe/p/bijoym1/AccentedASR/github/mtkd_rankweighted/vocab_files/ft_w2v2_base_960h", train_accent)
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    vocab_file_path = os.path.join(vocab_dir, 'vocab.json')
    if not os.path.isfile(vocab_dir):
        with open(vocab_file_path, 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)

    vocab_size = len(vocab_dict)
    
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file_path,
        unk_token="<unk>", 
        pad_token="<pad>", 
        word_delimiter_token="|"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=False
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )

    blank_token_id = processor.tokenizer.pad_token_id
    vocab_size = len(processor.tokenizer)
    print(blank_token_id)
    
    rand_int = random.randint(0, len(dataset["train"]))
    print("Target text:", dataset["train"][rand_int]["transcript"])
    print("Input array shape:", np.asarray(dataset["train"][rand_int]["audio"]["array"]).shape)
    print("Sampling rate:", dataset["train"][rand_int]["audio"]["sampling_rate"])

    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor)
    )
    
    saved_logits_path = "/scratch/elec/puhe/c/AESRC/logits/w2v2_base_960h/*"
    excluded_models = {"EDACCdevel", "AESRC200H"}

    all_dataset_logits_train = load_dataset_logits(train_accent, saved_logits_path, excluded_models)
    all_dataset_logits_devel = load_dataset_logits(devel_accent, saved_logits_path, excluded_models)
    all_dataset_logits_test = load_dataset_logits(test_accent, saved_logits_path, excluded_models)
    
    # ---------------------------------------------
    logits1 = []
    logits2 = []
    logits3 = []
    logits4 = []
    logits5 = []
    logits6 = []
    logits7 = []
    logits8 = []
    logits9 = []
    logits10 = []
    audio_paths = []
    transcripts = []
    accents = []
    
    input_values = []
    input_lengths = []
    labels = []

    num_samples = min(len(dataset['train']), len(all_dataset_logits_train[0]))
    for idx in tqdm(range(num_samples)):
    # for idx in tqdm(range(len(dataset['train']))):
        input_values.append(dataset['train'][idx]['input_values'])
        input_lengths.append(dataset['train'][idx]['input_length'])
        labels.append(dataset['train'][idx]['labels'])
        
        audio_paths.append(all_dataset_logits_train[0][idx]['audio_path'])
        transcripts.append(all_dataset_logits_train[0][idx]['transcript'])
        accents.append(all_dataset_logits_train[0][idx]['accent'])
        logits1.append(all_dataset_logits_train[0][idx]['logits'])
        logits2.append(all_dataset_logits_train[1][idx]['logits'])
        logits3.append(all_dataset_logits_train[2][idx]['logits'])
        logits4.append(all_dataset_logits_train[3][idx]['logits'])
        logits5.append(all_dataset_logits_train[4][idx]['logits'])
        logits6.append(all_dataset_logits_train[5][idx]['logits'])
        logits7.append(all_dataset_logits_train[6][idx]['logits'])
        logits8.append(all_dataset_logits_train[7][idx]['logits'])
        logits9.append(all_dataset_logits_train[8][idx]['logits'])
        logits10.append(all_dataset_logits_train[9][idx]['logits'])
        
    data_dict = {
        "audio_path": audio_paths,
        "transcript": transcripts,
        "accent": accents,
        "input_values": input_values,
        "input_length": input_lengths,
        "labels": labels,
        "logits1": logits1,
        "logits2": logits2,
        "logits3": logits3,
        "logits4": logits4,
        "logits5": logits5,
        "logits6": logits6,
        "logits7": logits7,
        "logits8": logits8,
        "logits9": logits9,
        "logits10": logits10,
    }
    all_dataset_logits_train_merged = Dataset.from_dict(data_dict)
    
    del all_dataset_logits_train
    gc.collect()
    
    # ---------------------------------------------
    
    logits1 = []
    logits2 = []
    logits3 = []
    logits4 = []
    logits5 = []
    logits6 = []
    logits7 = []
    logits8 = []
    logits9 = []
    logits10 = []
    audio_paths = []
    transcripts = []
    accents = []

    input_values = []
    input_lengths = []
    labels = []

    num_samples = min(len(dataset['devel']), len(all_dataset_logits_devel[0]))
    for idx in tqdm(range(num_samples)):
    # for idx in tqdm(range(len(dataset['devel']))):
        input_values.append(dataset['devel'][idx]['input_values'])
        input_lengths.append(dataset['devel'][idx]['input_length'])
        labels.append(dataset['devel'][idx]['labels'])
        
        audio_paths.append(all_dataset_logits_devel[0][idx]['audio_path'])
        transcripts.append(all_dataset_logits_devel[0][idx]['transcript'])
        accents.append(all_dataset_logits_devel[0][idx]['accent'])
        logits1.append(all_dataset_logits_devel[0][idx]['logits'])
        logits2.append(all_dataset_logits_devel[1][idx]['logits'])
        logits3.append(all_dataset_logits_devel[2][idx]['logits'])
        logits4.append(all_dataset_logits_devel[3][idx]['logits'])
        logits5.append(all_dataset_logits_devel[4][idx]['logits'])
        logits6.append(all_dataset_logits_devel[5][idx]['logits'])
        logits7.append(all_dataset_logits_devel[6][idx]['logits'])
        logits8.append(all_dataset_logits_devel[7][idx]['logits'])
        logits9.append(all_dataset_logits_devel[8][idx]['logits'])
        logits10.append(all_dataset_logits_devel[9][idx]['logits'])
        
    data_dict = {
        "audio_path": audio_paths,
        "transcript": transcripts,
        "accent": accents,
        "input_values": input_values,
        "input_length": input_lengths,
        "labels": labels,
        "logits1": logits1,
        "logits2": logits2,
        "logits3": logits3,
        "logits4": logits4,
        "logits5": logits5,
        "logits6": logits6,
        "logits7": logits7,
        "logits8": logits8,
        "logits9": logits9,
        "logits10": logits10,
    }
    all_dataset_logits_devel_merged = Dataset.from_dict(data_dict)
    
    del all_dataset_logits_devel
    gc.collect()
    
    # ---------------------------------------------
    
    logits1 = []
    logits2 = []
    logits3 = []
    logits4 = []
    logits5 = []
    logits6 = []
    logits7 = []
    logits8 = []
    logits9 = []
    logits10 = []
    audio_paths = []
    transcripts = []
    accents = []

    input_values = []
    input_lengths = []
    labels = []

    # for idx in tqdm(range(len(all_dataset_logits_test[0]))):
    num_samples = min(len(dataset['test']), len(all_dataset_logits_test[0]))
    for idx in tqdm(range(num_samples)):
    # for idx in tqdm(range(len(dataset['test']))):
        input_values.append(dataset['test'][idx]['input_values'])
        input_lengths.append(dataset['test'][idx]['input_length'])
        labels.append(dataset['test'][idx]['labels'])
        
        audio_paths.append(all_dataset_logits_test[0][idx]['audio_path'])
        transcripts.append(all_dataset_logits_test[0][idx]['transcript'])
        accents.append(all_dataset_logits_test[0][idx]['accent'])
        logits1.append(all_dataset_logits_test[0][idx]['logits'])
        logits2.append(all_dataset_logits_test[1][idx]['logits'])
        logits3.append(all_dataset_logits_test[2][idx]['logits'])
        logits4.append(all_dataset_logits_test[3][idx]['logits'])
        logits5.append(all_dataset_logits_test[4][idx]['logits'])
        logits6.append(all_dataset_logits_test[5][idx]['logits'])
        logits7.append(all_dataset_logits_test[6][idx]['logits'])
        logits8.append(all_dataset_logits_test[7][idx]['logits'])
        logits9.append(all_dataset_logits_test[8][idx]['logits'])
        logits10.append(all_dataset_logits_test[9][idx]['logits'])
        
    data_dict = {
        "audio_path": audio_paths,
        "transcript": transcripts,
        "accent": accents,
        "input_values": input_values,
        "input_length": input_lengths,
        "labels": labels,
        "logits1": logits1,
        "logits2": logits2,
        "logits3": logits3,
        "logits4": logits4,
        "logits5": logits5,
        "logits6": logits6,
        "logits7": logits7,
        "logits8": logits8,
        "logits9": logits9,
        "logits10": logits10,
    }
    all_dataset_logits_test_merged = Dataset.from_dict(data_dict)
    
    del all_dataset_logits_test
    gc.collect()
    
    # ---------------------------------------------
    
    final_dataset = DatasetDict({
        'train': all_dataset_logits_train_merged,
        'devel': all_dataset_logits_devel_merged,
        'test': all_dataset_logits_test_merged,
    })
    
    print(final_dataset)
    
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, 
        padding=True
    )
    
    # batch_size = 8
    
    train_dataloader = DataLoader(
        final_dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,  
    )

    devel_dataloader = DataLoader(
        final_dataset['devel'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,  
    )

    test_dataloader = DataLoader(
        final_dataset['test'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,  
    )
    
    
    # model_checkpoint_name = "facebook/wav2vec2-base-960h"
    
    config = Wav2Vec2Config.from_pretrained(
        model_checkpoint_name,
        vocab_size=vocab_size, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        model_checkpoint_name, 
        config=config, 
        ignore_mismatched_sizes=True
    )

    # ------------------------------------------------

    model.freeze_feature_encoder()
    model.to(device)

    # ------------------------------------------------

    loss_fn = LossWithLearnableWeights().to(device)

    optimizer = AdamW(
        [
            {"params": model.parameters(), "lr": learning_rate},  # Model parameters
            {"params": loss_fn.parameters(), "lr": learning_rate}  # Loss function parameters
        ]
    )


    num_training_steps = len(train_dataloader) * num_epochs

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_training_steps,
    )

    # ------------------------------------------------

    checkpoint_path = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/AccentedASR/github/mtkd_rankweighted/checkpoints/w2v2_base_960h_{train_accent}.pt"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        print(f"{'*'*50}\nModel Checkpoint has been loaded from '{checkpoint_path}'\n{'*'*50}")

    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    torch.cuda.empty_cache()

    # ------------------------------------------------
    if training:
        for epoch in range(start_epoch, num_epochs):
            history = train_mtkd_v2(
                model, checkpoint_path, 
                processor, loss_fn, train_dataloader, optimizer, lr_scheduler, 
                epoch, wer_metric, cer_metric, device
            )
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Total Loss: {history['total_loss'][-1]:.4f} | "
                f"WER: {history['wer'][-1]:.4f} | "
                f"CER: {history['cer'][-1]:.4f}")
            
        print(f"\n\n{'*'*50}\nHistory:\n{'*'*50}\n{history}\n{'*'*50}\n\n")    

        # ------------------------------------------------
        
        devel_loss, devel_wer, devel_cer = devel(
            model, processor, devel_dataloader, wer_metric, cer_metric, device
        )
        print(f"Devel CTC Loss: {devel_loss:.4f} | WER: {devel_wer:.4f} | CER: {devel_cer:.4f}")
        
        # ------------------------------------------------
        
    else:
        test_loss, test_wer, test_cer, test_predictions, test_true_labels = test(
            model, processor, test_dataloader, wer_metric, cer_metric, device
        )
        
        print(len(test_true_labels), len(test_predictions))
        print(f"\n{'*'*50}\nTest CTC Loss: {test_loss:.4f} | WER: {test_wer*100:.4f} | CER: {test_cer*100:.4f}\n{'*'*50}\n")

        csv_path = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/AccentedASR/github/mtkd_avg/csv_files/mtkd_rankweighted/w2v2_base_960h_{train_accent}_test_{test_accent}.csv"
        df = pd.DataFrame({"transcription": test_true_labels, "prediction": test_predictions})
        df.to_csv(csv_path, index=False)
        
        
    # ------------------------------------------------
        
    
    
if __name__ == "__main__":
    main()
    
    