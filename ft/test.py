import torch
from tqdm import tqdm

def test(model, processor, dataloader, wer_metric, cer_metric, num_samples, device):
    model.eval()

    all_pred_str = []
    all_label_str = []
    all_accent_str = []

    for batch in tqdm(dataloader):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        
        accents = batch["accents"]
        all_accent_str.extend(accents)

        with torch.no_grad():
            outputs = model(input_values)

        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)
        all_pred_str.extend(pred_str)
        
        label_ids = labels.detach().cpu().numpy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        all_label_str.extend(label_str)
    
    all_label_str_filtered = []
    all_pred_str_filtered = []
    all_accent_str_filtered = []
    for ref, pred, acnt in zip(all_label_str, all_pred_str, all_accent_str):
        if ref.strip():
            all_label_str_filtered.append(ref)
            all_pred_str_filtered.append(pred)
            all_accent_str_filtered.append(acnt)
    
    # epoch_wer = wer_metric.compute(predictions=all_pred_str, references=all_label_str)
    # epoch_cer = cer_metric.compute(predictions=all_pred_str, references=all_label_str)
    epoch_wer = wer_metric.compute(predictions=all_pred_str_filtered, references=all_label_str_filtered)
    epoch_cer = cer_metric.compute(predictions=all_pred_str_filtered, references=all_label_str_filtered)
    
    print("-" * 50)
    print(f"Test WER: {epoch_wer:.4f}, Test CER: {epoch_cer:.4f}")
    print("-" * 50)
    for i, (true, pred) in enumerate(zip(all_label_str_filtered, all_pred_str_filtered)):
        if i >= num_samples: 
            break
        print(f"Sample {i + 1}")
        print(f"Correct Transcription  : {true}")
        print(f"Predicted Transcription: {pred}")
        print("-" * 50)
        
    return (all_label_str_filtered, all_pred_str_filtered, all_accent_str_filtered)