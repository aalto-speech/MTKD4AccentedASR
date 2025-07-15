from tqdm import tqdm
import torch
from datasets import Dataset


def extract_logits(model, dataloader, train_accent, test_accent, foundation_model, device):
    model.eval()
    all_input_values = []
    all_labels = []
    all_accents = []
    all_audio_paths = []
    all_transcripts = []
    all_audios = []
    all_input_lengths = []
    all_logits = []

    for batch in tqdm(dataloader):
        input_values = batch["input_values"]
        labels = batch["labels"]
        accents = batch["accents"]
        audio_paths = batch["audio_paths"]
        transcripts = batch["transcripts"]
        audios = batch["audios"]
        input_lengths = batch["input_lengths"]
        
        with torch.no_grad():
            outputs = model(input_values.to(device))
            
        logits = outputs.logits
        logits_normalized = torch.nn.functional.softmax(logits, dim=-1)
        
        
        all_logits.extend(logits_normalized)
        all_input_values.extend(input_values)
        all_labels.extend(labels)
        all_accents.extend(accents)
        all_audio_paths.extend(audio_paths)
        all_transcripts.extend(transcripts)
        all_audios.extend(audios)
        all_input_lengths.extend(input_lengths)
        
    data_dict = {
        "audio_path": all_audio_paths,
        "transcript": all_transcripts,
        "accent": all_accents,
        "logits": all_logits,
    }
    new_dataset = Dataset.from_dict(data_dict)
    # dir_name = f"/scratch/elec/puhe/c/AESRC/logits/{foundation_model}/ft_{train_accent}/w2v2{train_accent}_{test_accent}"
    # os.makedirs(dir_name, exist_ok=True)
    # new_dataset.save_to_disk(dir_name)
    return new_dataset