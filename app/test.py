import os
import json
import re
import sys
from processing.speech_text_whisper import transcribe_audio, improve_transcript, save_transcript
import random
import json
from collections import defaultdict


import numpy as np

def word_error_rate(reference, hypothesis):
    """
    Calculate the Word Error Rate (WER) between the ground truth (reference) and the predicted transcript (hypothesis).
    """
    reference = reference.split()
    hypothesis = hypothesis.split()

    # Using dynamic programming to calculate the Levenshtein distance
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1))

    for i in range(len(reference) + 1):
        for j in range(len(hypothesis) + 1):
            if i == 0:
                d[i][j] = j
            elif j == 0:
                d[i][j] = i
            else:
                cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(reference)][len(hypothesis)] / float(len(reference))

def char_error_rate(reference, hypothesis):
    """
    Calculate the Character Error Rate (CER) between the ground truth (reference) and the predicted transcript (hypothesis).
    """
    reference = reference.replace(" ", "")  # Remove spaces for character level comparison
    hypothesis = hypothesis.replace(" ", "")

    # Using dynamic programming to calculate the Levenshtein distance
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1))

    for i in range(len(reference) + 1):
        for j in range(len(hypothesis) + 1):
            if i == 0:
                d[i][j] = j
            elif j == 0:
                d[i][j] = i
            else:
                cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(reference)][len(hypothesis)] / float(len(reference))


def calculate_overall_metric(ground_truths, transcriptions, metric_func):
    total_metric = 0
    for gt, trans in zip(ground_truths, transcriptions):
        total_metric += metric_func(gt, trans)
    return total_metric / len(ground_truths)


def transpose_json(input_file, output_file):
    # Load the input JSON file
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    # Create a defaultdict to store the transposed structure
    transposed = defaultdict(dict)

    # Iterate through the original data to fill the transposed dictionary
    for entry in data:
        for key, value in entry.items():
            if key != 'id' and key != 'thema':  # Skip 'id' and 'thema' fields
                transposed[key][entry['id']] = value
    
    # Convert defaultdict back to a regular dict
    transposed = dict(transposed)

    # Save the transposed structure to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(transposed, outfile, ensure_ascii=False, indent=4)

def get_transcript_from_file(file_name, transposed_data):
    # Extract the language and ID from the file name using regex
    match = re.match(r"ch_([a-z]+)_(\d+)\.wav", file_name)
    if match:
        language = f"ch_{match.group(1)}"   # Extract language (e.g., 'vs')
        file_id = str(int(match.group(2)))  # Extract id (e.g., '0861')
        
        # Check if the language exists in the transposed data
        if language in transposed_data:
            # Get the transcript for the given id
            language_data = transposed_data[language]
            language_data_de = transposed_data['de']
            if file_id in language_data:
                return language_data[file_id], language_data_de[file_id]
            else:
                return f"Transcript not found for id {file_id}."
        else:
            return f"Language {language} not found in the transposed data."
    else:
        return "File name format is incorrect."



# Set random seed if you want reproducibility
random.seed(42)

# Paths
test_files_path = 'data/data1.1/' 

# Preprocess transcripts
transpose_json(f'{test_files_path}sentences_ch_de_transcribed.json', f'{test_files_path}sentences_ch_de_transcribed_t.json')
transcripts = json.load(open(f'{test_files_path}sentences_ch_de_transcribed_t.json', 'r'))
print(f"Loaded {len(transcripts)} transcripts.")

# List folders
test_folders = os.listdir(test_files_path)
print(f"Found {test_folders} folders in the test directory.")

# Collect all .wav files
all_files = []
for folder in test_folders:
    folder_path = os.path.join(test_files_path, folder)
    if os.path.isdir(folder_path):
        test_files = os.listdir(folder_path)
        for file in test_files:
            if file.endswith('.wav'):
                all_files.append((folder, file))

print(f"Total .wav files found: {len(all_files)}")

# Randomly select 10 files
sampled_files = random.sample(all_files, 10)
print(f"Randomly selected {len(sampled_files)} files for testing.")

test_results = []

# Process sampled files
for folder, file in sampled_files:
    print(f"Processing file: {file} in folder: {folder}")
    file_path = os.path.join(test_files_path, folder, file)
    
    ground_truth, ground_truth_de = get_transcript_from_file(file, transcripts)
    print(f"Ground Truth: {ground_truth}")
    print(f"Ground Truth DE: {ground_truth_de}")
    
    # Transcribe
    transcript = transcribe_audio(file_path)
    print(f"Transcript: {transcript}")
    
    # Optionally improve
    improved_transcript = transcript  # improve_transcript(transcript, language_hint="Swiss German")
    print(f"Improved Transcript: {improved_transcript}")

    # Compute WER and CER
    wer = word_error_rate(ground_truth_de, improved_transcript)
    cer = char_error_rate(ground_truth_de, improved_transcript)
    
    test_results.append({
        "folder": folder,
        "file": file,
        "ground_truth": ground_truth_de,
        "transcription": improved_transcript,
        "wer": wer,
        "cer": cer
    })

    print(f"Word Error Rate (WER): {wer}")
    print(f"Character Error Rate (CER): {cer}")

# Calculate overall metrics from ground truths and transcriptions
overall_wer = word_error_rate(
    " ".join([result["ground_truth"] for result in test_results]),
    " ".join([result["transcription"] for result in test_results])
)
overall_cer = char_error_rate(
    " ".join([result["ground_truth"] for result in test_results]),
    " ".join([result["transcription"] for result in test_results])
)

print(f"Overall WER: {overall_wer}")
print(f"Overall CER: {overall_cer}")
