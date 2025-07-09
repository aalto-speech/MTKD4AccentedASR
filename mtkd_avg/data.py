import re


def remove_special_characters(batch):
    """
    Cleans the 'transcript' field in a dataset batch by:
    - Converting all text to uppercase.
    - Removing a set of special characters (commas, question marks, punctuation, symbols, etc.).
    - Stripping leading/trailing whitespace.
    - Appending a trailing space to the cleaned transcript.

    Args:
        batch (dict): A single example from the dataset with at least a 'transcript' key.

    Returns:
        dict: The modified batch with the cleaned transcript.
    """
    
    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"<>\_\|\…]'
    text = batch["transcript"].upper()
    text = text.replace("\ufeff", "")
    batch["transcript"] = re.sub(chars_to_ignore_regex, '', text).strip() + " "
    return batch





def clean_transcript(example):
    """
    Normalizes and sanitizes the 'transcript' field in a dataset example by:
    - Converting text to lowercase.
    - Replacing accented or non-standard characters with basic ASCII equivalents.
    - Removing annotation tokens commonly used in speech corpora (e.g., <laugh>, <breath>).
    - Stripping any leading/trailing whitespace.

    Args:
        example (dict): A single example from the dataset containing a 'transcript' key.

    Returns:
        dict: The modified example with a cleaned and normalized transcript.
    """
    
    unique_tokens = ['<overlap>','<laugh>','<dtmf>','<foreign>','<breath>','<cough>','<lipsmack>','<ring>','<click>']
    cleaned_text = example["transcript"].lower()
    
    char_mapping = { 'à': 'a', 'ä': 'a', 'ç': 'c', 'é': 'e', 'í': 'i', 'ñ': 'n', 'ó': 'o', 'ô': 'o', 'ù': 'u', 'ú': 'u', 'ü': 'u', 'đ': 'd', 'ı': 'i', 'ạ': 'a', 'ả': 'a', 'ậ': 'a', 'ắ': 'a', 'ế': 'e', 'ệ': 'e', 'ồ': 'o'}
    
    cleaned_text = ''.join(char_mapping.get(char, char) for char in cleaned_text)
    
    for u_token in unique_tokens:
        cleaned_text = cleaned_text.replace(u_token, '') 
    cleaned_text = cleaned_text.strip() 
    
    return {"transcript": cleaned_text}






def extract_all_chars(batch):
    """
    Extracts all unique characters from a batch of transcripts to help build a vocabulary.

    This function:
    - Joins all transcripts into one long string.
    - Extracts the set of unique characters.
    - Returns both the full concatenated text and the list of unique characters.

    Args:
        batch (dict): A batch of examples with a "transcript" key, where each value is a string.

    Returns:
        dict: A dictionary containing:
            - "vocab": A list with a single element (the list of unique characters).
            - "all_text": A list with a single element (the full concatenated text).
    """
    
    all_text = " ".join(batch["transcript"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}






def prepare_dataset(batch, processor):
    """
    Preprocesses a batch of audio-transcription pairs for use in speech recognition training.

    This function:
    - Extracts the raw audio array and sampling rate from the batch.
    - Uses the processor to convert the audio into input values (features for the model).
    - Calculates and stores the length of the input values.
    - Converts the corresponding transcript into label IDs using the processor in target mode.

    Args:
        batch (dict): A dictionary containing:
            - "audio": A dictionary with keys "array" (the waveform) and "sampling_rate".
            - "transcript": A string representing the transcription of the audio.
        processor: A processor object (e.g., from Hugging Face's `Wav2Vec2Processor`) 
                   that handles feature extraction and tokenization.

    Returns:
        dict: The original batch, updated with:
            - "input_values": Model-ready input features derived from the audio.
            - "input_length": The length of the input sequence.
            - "labels": Tokenized label IDs derived from the transcript.
    """
    
    audio = batch["audio"]

    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    return batch



