import ahocorasick
from datasets import load_dataset
import pandas as pd


dataset = load_dataset("allenai/scirepeval", 'biomimicry')

words_to_match = []
with open('our_prepared_words.txt', 'r') as file:
    words_to_match = [line.strip().lower() for line in file]  

ANA_PHRASE_AHO = ahocorasick.Automaton()

for idx, word in enumerate(words_to_match):
    ANA_PHRASE_AHO.add_word(word, (idx, word))

ANA_PHRASE_AHO.make_automaton()

matched_data = []

for record in dataset['evaluation']:
    abstract = record['abstract']
    if abstract:
        word_counts = {}
    
        for end_index, (_, found) in ANA_PHRASE_AHO.iter(abstract.lower()):
            
            start_index = end_index - len(found) + 1
            if (start_index == 0 or not abstract[start_index - 1].isalnum()) and (end_index + 1 == len(abstract) or not abstract[end_index + 1].isalnum()):
                word_counts[found] = word_counts.get(found, 0) + 1

        if word_counts:
            record_data = {
                'doc_id': record['doc_id'],
                'doi': record['doi'],
                'corpus_id': record['corpus_id'],
                'title': record['title'],
                'label': record['label'],
                'venue': record['venue'],
                'Abstract': abstract,
                'Word Counts': word_counts
            }
            matched_data.append(record_data)


matched_df = pd.DataFrame(matched_data)
