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

from data import remove_special_characters, clean_transcript, prepare_dataset
from train import train
from devel import devel
from logits import extract_logits
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
    foundation_model = "w2v2_base_960h"
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
    
    dataset = DatasetDict({
        'train': dataset_train,
        'devel': dataset_devel,
        'test': dataset_test,
    })
    dataset = dataset.map(remove_special_characters)
    print(dataset)
    
    del dataset_train
    del dataset_devel
    del dataset_test
    
    vocab_dict = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}
    
    
    vocab_dir = os.path.join("/m/triton/scratch/elec/t405-puhe/p/bijoym1/AccentedASR/github/logits/vocab_files/ft_w2v2_base_960h", train_accent)
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

    # Check if pad_token_id is within the valid range for CTC
    if blank_token_id >= vocab_size:
        raise ValueError(f"pad_token_id ({blank_token_id}) is outside the valid range of vocabulary indices (0 to {vocab_size - 1})")

    
    rand_int = random.randint(0, len(dataset["train"]))
    print("Target text:", dataset["train"][rand_int]["transcript"])
    print("Input array shape:", np.asarray(dataset["train"][rand_int]["audio"]["array"]).shape)
    print("Sampling rate:", dataset["train"][rand_int]["audio"]["sampling_rate"])

    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor), 
        # remove_columns=dataset.column_names["train"], 
    )
    
    max_input_length_in_sec = 10.0
    dataset["train"] = dataset["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
    dataset["devel"] = dataset["devel"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
    max_input_length_in_sec = 30.0
    dataset["test"] = dataset["test"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
    print(dataset)
    
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, 
        padding=True
    )

    train_dataloader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,  
    )

    devel_dataloader = DataLoader(
        dataset['devel'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,  
    )

    test_dataloader = DataLoader(
        dataset['test'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,  
    )
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # obokkkk/wav2vec2-base-960h-finetuned_common_voice

    
    checkpoint_path = f"/m/triton/scratch/elec/t405-puhe/p/bijoym1/AccentedASR/github/logits/checkpoints/ft_w2v2_base_960h_{train_accent}.pt"
    
    if os.path.exists(checkpoint_path):
        model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path)
        print(f"Model has been loaded from checkpoint at {checkpoint_path}")
    else:
        # "facebook/wav2vec2-xls-r-300m" "facebook/wav2vec2-base-100k-voxpopuli" "facebook/wav2vec2-base"
        # "patrickvonplaten/wav2vec2-base-timit-demo-colab" "AKulk/wav2vec2-base-timit-epochs20"
        # model_checkpoint_name = "patrickvonplaten/wav2vec2-base-timit-demo-colab"
        # model_checkpoint_name = "facebook/wav2vec2-base-960h"
        config = Wav2Vec2Config.from_pretrained(
            model_checkpoint_name,
            vocab_size=vocab_size, 
            ctc_loss_reduction="mean", 
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        # model = Wav2Vec2ForCTC.from_pretrained(model_checkpoint_name, config=config)
        model = Wav2Vec2ForCTC.from_pretrained(model_checkpoint_name, config=config, ignore_mismatched_sizes=True)
        
    # print(model)
    
    model.freeze_feature_encoder()
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    num_training_steps = len(train_dataloader) * num_epochs

    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=1000, 
        num_training_steps=num_training_steps
    )
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch: {epoch+1}")
        train_loss, train_wer, train_cer = train(
            model, optimizer, lr_scheduler, processor, train_dataloader, wer_metric, cer_metric, device
        )
        
        devel_loss, devel_wer, devel_cer = devel(
            model, processor, devel_dataloader, wer_metric, cer_metric, device
        )
        
        print(f"Train Loss: {train_loss:.4f} | WER: {train_wer:.4f} | CER: {train_cer:.4f}")
        print(f"Devel Loss: {devel_loss:.4f} | WER: {devel_wer:.4f} | CER: {devel_cer:.4f}")
        
        if best_loss > train_loss:
            print("-" * 50)
            print(f"Train loss improved from {best_loss:.4f} to {train_loss:.4f}. Saving model checkpoint...")
            best_loss = train_loss
            model.save_pretrained(checkpoint_path)
            print("-" * 50)
        
        
    
    new_dataset = extract_logits(model, test_dataloader, train_accent, test_accent, foundation_model, device)
    dir_name = f"/scratch/elec/puhe/c/AESRC/logits/{foundation_model}/ft_on_{train_accent}_Accent/tested_on_{test_accent}"
    os.makedirs(dir_name, exist_ok=True)
    new_dataset.save_to_disk(dir_name)
    
    
    
if __name__ == "__main__":
    main()
    
    