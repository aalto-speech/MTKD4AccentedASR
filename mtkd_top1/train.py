import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn


def train(model, optimizer, lr_scheduler, processor, dataloader, wer_metric, cer_metric, device):  
    """
    Trains a speech recognition model for one epoch.

    This function:
    - Performs a forward and backward pass for each batch.
    - Optimizes the model using the provided optimizer and learning rate scheduler.
    - Decodes predictions and labels to compute Word Error Rate (WER) and Character Error Rate (CER).

    Args:
        model (torch.nn.Module): The speech recognition model to train.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        lr_scheduler: A learning rate scheduler (e.g., from Hugging Face or PyTorch).
        processor: A processor object for feature extraction and decoding (e.g., `Wav2Vec2Processor`).
        dataloader (torch.utils.data.DataLoader): A DataLoader providing training batches.
        wer_metric: A metric object with a `.compute()` method to calculate WER.
        cer_metric: A metric object with a `.compute()` method to calculate CER.
        device (torch.device): The device to move tensors and the model to (CPU or GPU).

    Returns:
        tuple:
            - epoch_loss (float): The average loss over the epoch.
            - epoch_wer (float): Word Error Rate over the epoch.
            - epoch_cer (float): Character Error Rate over the epoch.
    """
    
    model.train()
    total_loss = 0
    
    all_pred_str = []
    all_label_str = []

    # Wrap DataLoader with tqdm
    for batch in tqdm(dataloader, desc="Training", leave=True, total=len(dataloader)):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

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





def train_mtkd(model, checkpoint_path, processor, loss_fn, train_dataloader, optimizer, lr_scheduler, epoch, wer_metric, cer_metric, device):
    """
    Trains a speech recognition model using Multi-Teacher Knowledge Distillation (MTKD) for one epoch.

    This function:
    - Performs forward and backward passes using both the main model and teacher ensemble logits.
    - Computes a custom loss combining:
        - CTC loss between model outputs and ground truth labels.
        - Kullback-Leibler (KL) divergence between model logits and averaged teacher logits.
    - Uses learnable weights (alpha and beta) to balance the loss components.
    - Collects and logs training statistics including WER and CER.
    - Saves training state to a checkpoint.

    Args:
        model (torch.nn.Module): The student ASR model to be trained.
        checkpoint_path (str): File path where training checkpoint will be saved.
        processor: Hugging Face processor (e.g., `Wav2Vec2Processor`) for decoding predictions and labels.
        loss_fn (nn.Module): Custom loss function that combines CTC and KL-divergence with learnable weights.
        train_dataloader (DataLoader): Dataloader yielding batches with inputs, labels, and teacher logits.
        optimizer (torch.optim.Optimizer): Optimizer to update model weights.
        lr_scheduler: Learning rate scheduler.
        epoch (int): Current epoch number (used for logging and checkpointing).
        wer_metric: Metric object with `.compute()` method to calculate Word Error Rate (WER).
        cer_metric: Metric object with `.compute()` method to calculate Character Error Rate (CER).
        device (torch.device): Device (CPU or GPU) to run the training on.

    Returns:
        dict: A history dictionary containing:
            - 'wer': List of WER scores for each epoch.
            - 'cer': List of CER scores for each epoch.
            - 'alpha': List of alpha (CTC loss weight) values per epoch.
            - 'beta': List of beta (KL divergence weight) values per epoch.
            - 'ctc': List of CTC losses per epoch.
            - 'kld': List of KL divergence losses per epoch.
            - 'total_loss': List of combined total losses per epoch.
    """
    
    # ----------------------------------------
    
    model.train()
    loss_fn.train()
    total_epoch_loss = .0
    
    # ----------------------------------------
    
    history = {
        'wer': [],
        'cer': [],
        'alpha': [],
        'beta': [],
        'ctc': [],
        'kld': [],
        'total_loss': []
    }
    
    # ----------------------------------------
    
    all_pred_str = []
    all_label_str = []
    
    # ----------------------------------------

    # Wrap the DataLoader with tqdm to track training progress
    progress_bar = tqdm(enumerate(train_dataloader), 
                        desc=f"Training Epoch {epoch+1}", 
                        leave=True, 
                        total=len(train_dataloader))

    # ----------------------------------------
    
    for idx, batch in progress_bar:
        batch['avg_logits'] = torch.tensor(batch['avg_logits']).permute(1, 0, 2).to(device)
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].squeeze(0).to(device)
        optimizer.zero_grad()

        # Forward Pass: ASR Model
        outputs = model(input_values, labels=labels)
        logits = outputs.logits
        
        # ----------------------------------------
        
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)
        all_pred_str.extend(pred_str)
        label_ids = labels.detach().cpu().numpy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        all_label_str.extend(label_str)
        
        # ----------------------------------------
        
        logits = logits.permute(1, 0, 2)  # (seq_len, batch, vocab)
        avg_logits = batch['avg_logits'].to(device)

        labels = labels.clone()
        
        # ****************************************
        # if labels.ndim == 1:
        #     labels = labels.unsqueeze(1)
        # ****************************************
        
        # Count non-padding values
        target_lengths = (labels != -100).sum(dim=1)  
        
        # Max input length for each sample
        input_lengths = torch.full(
            (logits.shape[1],),  # batch_size
            logits.shape[0],  # max_seq_length from logits
            dtype=torch.long,
            device=device
        )

        ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean')
        
        ctc_loss = ctc_loss_fn(
            F.log_softmax(logits, dim=-1), 
            labels, 
            input_lengths, 
            target_lengths
        )

        # Combine Losses using Learnable Weights
        total_loss, alpha, beta, ctc_loss, kl_loss = loss_fn(ctc_loss, logits, avg_logits)

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_epoch_loss += total_loss.item()

        # Update tqdm progress bar with loss values
        progress_bar.set_postfix({
            "Loss": f"{total_loss.item():.4f}",
            "Alpha": f"{alpha.item():.4f}",
            "Beta": f"{beta.item():.4f}",
            "CTC": f"{ctc_loss.item():.4f}",
            "KLD": f"{kl_loss.item():.4f}"
        })

    avg_epoch_loss = total_epoch_loss / len(train_dataloader)
    
    # ----------------------------------------
    
    all_label_str_filtered = []
    all_pred_str_filtered = []
    for ref, pred in zip(all_label_str, all_pred_str):
        if ref.strip():
            all_label_str_filtered.append(ref)
            all_pred_str_filtered.append(pred)
    epoch_wer = wer_metric.compute(predictions=all_pred_str_filtered, references=all_label_str_filtered)
    epoch_cer = cer_metric.compute(predictions=all_pred_str_filtered, references=all_label_str_filtered)
    
    # ----------------------------------------
    
    history['wer'].append(epoch_wer)
    history['cer'].append(epoch_cer)
    history['alpha'].append(alpha.item())
    history['beta'].append(beta.item())
    history['ctc'].append(ctc_loss.item())
    history['kld'].append(kl_loss.item())
    history['total_loss'].append(avg_epoch_loss)
    
    # ----------------------------------------
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
        'history': history
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # ----------------------------------------
    
    return history

    # ----------------------------------------
    
    
    
    
    
def train_mtkd_v2(model, checkpoint_path, processor, loss_fn, train_dataloader, optimizer, lr_scheduler, epoch, wer_metric, cer_metric, device):
    """
    Train a speech model using Multi-Teacher Knowledge Distillation (MT-KD) with learnable loss weights.

    Args:
        model (nn.Module): The student ASR model.
        checkpoint_path (str): Path to save the training checkpoint.
        processor (Wav2Vec2Processor): HuggingFace processor for encoding/decoding audio and text.
        loss_fn (nn.Module): Custom loss module that combines CTC and KL loss with learnable weights.
        train_dataloader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        epoch (int): Current epoch number.
        wer_metric (datasets.Metric): WER metric tracker.
        cer_metric (datasets.Metric): CER metric tracker.
        device (torch.device): Torch device (e.g., "cuda" or "cpu").

    Returns:
        dict: Training history with WER, CER, and loss values.
    """
    
    # ----------------------------------------
    
    model.train()
    loss_fn.train()
    total_epoch_loss = .0
    
    # ----------------------------------------
    
    history = {
        'wer': [],
        'cer': [],
        'alpha': [],
        'beta': [],
        'ctc': [],
        'kld': [],
        'total_loss': []
    }
    
    # ----------------------------------------
    
    all_pred_str = []
    all_label_str = []
    
    # ----------------------------------------

    # Wrap the DataLoader with tqdm to track training progress
    progress_bar = tqdm(enumerate(train_dataloader), 
                        desc=f"Training Epoch {epoch+1}", 
                        leave=True, 
                        total=len(train_dataloader))
    
    # ----------------------------------------
    
    for idx, batch in progress_bar:
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].squeeze(0).to(device)
        
        optimizer.zero_grad()

        # Forward Pass: ASR Model
        outputs = model(input_values, labels=labels)
        logits = outputs.logits
        
        # ----------------------------------------
        
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)
        all_pred_str.extend(pred_str)
        label_ids = labels.detach().cpu().numpy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        all_label_str.extend(label_str)
        
        # ----------------------------------------
        
        logits = logits.permute(1, 0, 2)  # (seq_len, batch, vocab)
        
        # ----------------------------------------
        
        all_logits = []
        for idx in range(1, 11):
            all_logits.append(batch[f'logits{idx}'].to(device).permute(1, 0, 2))
        
        # ----------------------------------------
        
        labels = labels.clone()
        
        # Count non-padding values
        target_lengths = (labels != -100).sum(dim=1)  
        
        # Max input length for each sample
        input_lengths = torch.full(
            (logits.shape[1],),  # batch_size
            logits.shape[0],  # max_seq_length from logits
            dtype=torch.long,
            device=device
        )

        ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean')
        
        ctc_loss = ctc_loss_fn(
            F.log_softmax(logits, dim=-1), 
            labels, 
            input_lengths, 
            target_lengths
        )

        # Combine Losses using Learnable Weights
        total_loss, alpha, beta, kl_weights, ctc_loss, kl_loss = loss_fn(ctc_loss, logits, all_logits)


        # Backpropagation
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_epoch_loss += total_loss.item()

        # Update tqdm progress bar with loss values
        progress_bar.set_postfix({
            "Loss": f"{total_loss.item():.4f}",
            "Alpha": f"{alpha.item():.4f}",
            "Beta": f"{beta.item():.4f}",
            "CTC": f"{ctc_loss.item():.4f}",
            "KLD": f"{kl_loss.item():.4f}",
            # "T1": f"{kl_weights[0].item():.2f}",
            # "T2": f"{kl_weights[1].item():.2f}",
            # "T3": f"{kl_weights[2].item():.2f}",
            # "T4": f"{kl_weights[3].item():.2f}",
            # "T5": f"{kl_weights[4].item():.2f}",
            # "T6": f"{kl_weights[5].item():.2f}",
            # "T7": f"{kl_weights[6].item():.2f}",
            # "T8": f"{kl_weights[7].item():.2f}",
            # "T9": f"{kl_weights[8].item():.2f}",
            # "T10": f"{kl_weights[9].item():.2f}",
        })
        
    # print(f"Training Epoch {epoch+1} | T[1]: {kl_weights[0].item():.2f}")
    print(f"Training Epoch {epoch+1} | " + " | ".join([f"T[{i+1}]: {kl_weights[i].item():.2f}" for i in range(10)]))
    print("\n")
    
    avg_epoch_loss = total_epoch_loss / len(train_dataloader)
    
    # ----------------------------------------
    
    all_label_str_filtered = []
    all_pred_str_filtered = []
    for ref, pred in zip(all_label_str, all_pred_str):
        if ref.strip():
            all_label_str_filtered.append(ref)
            all_pred_str_filtered.append(pred)
    epoch_wer = wer_metric.compute(predictions=all_pred_str_filtered, references=all_label_str_filtered)
    epoch_cer = cer_metric.compute(predictions=all_pred_str_filtered, references=all_label_str_filtered)
    
    # ----------------------------------------
    
    history['wer'].append(epoch_wer)
    history['cer'].append(epoch_cer)
    history['alpha'].append(alpha.item())
    history['beta'].append(beta.item())
    history['ctc'].append(ctc_loss.item())
    history['kld'].append(kl_loss.item())
    history['total_loss'].append(avg_epoch_loss)
    
    # ----------------------------------------
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
        'history': history
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # ----------------------------------------
    
    return history

    # ----------------------------------------
