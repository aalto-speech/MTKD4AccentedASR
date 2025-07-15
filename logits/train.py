import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn



def train(model, optimizer, lr_scheduler, processor, dataloader, wer_metric, cer_metric, device):  
    model.train()
    total_loss = 0
    
    all_pred_str = []
    all_label_str = []
    # all_accent_str = []

    for batch in tqdm(dataloader):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        
        # accents = batch["accents"]
        # all_accent_str.extend(accents)
        
        optimizer.zero_grad()
        
        outputs = model(input_values, labels=labels)
        
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)
        all_pred_str.extend(pred_str)
        
        label_ids = labels.detach().cpu().numpy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        all_label_str.extend(label_str)
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        
    epoch_loss = total_loss / len(dataloader)
    
    all_label_str_filtered = []
    all_pred_str_filtered = []
    for ref, pred in zip(all_label_str, all_pred_str):
        if ref.strip():
            all_label_str_filtered.append(ref)
            all_pred_str_filtered.append(pred)
    
    epoch_wer = wer_metric.compute(predictions=all_pred_str_filtered, references=all_label_str_filtered)
    epoch_cer = cer_metric.compute(predictions=all_pred_str_filtered, references=all_label_str_filtered)
    
    return epoch_loss, epoch_wer, epoch_cer


