import torch
from tqdm import tqdm


def test(model, processor, dataloader, wer_metric, cer_metric, device):  
    """
    Evaluates a trained speech recognition model on a test dataset.

    This function:
    - Runs inference on the test set without updating gradients.
    - Computes average loss, Word Error Rate (WER), and Character Error Rate (CER).
    - Also returns the full list of non-empty predicted and reference transcriptions for further analysis.

    Args:
        model (torch.nn.Module): The trained ASR model to evaluate.
        processor: A processor object (e.g., `Wav2Vec2Processor`) used for decoding predicted and true labels.
        dataloader (DataLoader): A DataLoader providing test batches with input values and labels.
        wer_metric: A metric object with `.compute()` method to calculate Word Error Rate.
        cer_metric: A metric object with `.compute()` method to calculate Character Error Rate.
        device (torch.device): The device (CPU or GPU) used for computation.

    Returns:
        tuple:
            - epoch_loss (float): Average test loss over the dataset.
            - epoch_wer (float): Word Error Rate for the test set.
            - epoch_cer (float): Character Error Rate for the test set.
            - all_pred_str_filtered (list[str]): List of non-empty predicted transcriptions.
            - all_label_str_filtered (list[str]): List of corresponding reference transcriptions.
    """
    
    model.eval()
    total_loss = 0
    
    all_pred_str = []
    all_label_str = []

    # Wrap DataLoader with tqdm
    for batch in tqdm(dataloader, desc="Validation", leave=True, total=len(dataloader)):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.no_grad():
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
    
    return epoch_loss, epoch_wer, epoch_cer, all_pred_str_filtered, all_label_str_filtered





