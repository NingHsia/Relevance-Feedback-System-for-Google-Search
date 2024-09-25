# also consider non-html files 
# reorder too slow if consider all permutations -> order of original query remain the same
# repeated words in query? -> no



import pprint
import sys
from googleapiclient.discovery import build
from collections import Counter
import re
from math import log, log2

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk import FreqDist

from copy import deepcopy


def getResultFromGoogle(query, API_key, search_engine_id):
    service = build(
        "customsearch", "v1", developerKey=API_key
    )

    res = (
        service.cse()
        .list(
            q=query,
            cx=search_engine_id,
            lr="lang_en"
        )
        .execute()
    )

    if 'items' in res:
        results = []
        for item in res['items']:
            # print(item)
            if 'fileFormat' in item: 
                results.append({"URL": item['link'], "Title": item['title'], "Summary": item['snippet'], "fileFormat": item["fileFormat"]})
            else:
                results.append({"URL": item['link'], "Title": item['title'], "Summary": item['snippet']})

        return results
    else:
        return None
    
def get_verb_set(document):
    title, snippet= document["Title"].lower(),document["Summary"].lower()
    verb_set = Counter(re.split('[^a-zA-Z]+', title+" "+snippet))
    return verb_set

def calculate_scores(docs, relevent_index, stopwords_set, query):
    irrelevent_index = []
    for i in range(len(docs)):
      if i not in relevent_index:
        irrelevent_index.append(i)
    verb_sets = [get_verb_set(doc) for doc in docs]
    candidate_terms = set()
    for i in relevent_index:
        candidate_terms.update(verb_sets[i].keys())
    # stopword elimination
    candidate_terms -= stopwords_set
    # no repeat query words
    query_word_set = set(query.split())
    candidate_terms -= query_word_set

    results = []
    for t in candidate_terms:
        term_occurence_count = sum([verb_sets[i][t] if t in verb_sets[i] else 0 for i in relevent_index])
        document_occurence_count = sum([1 if t in verb_sets[i] else 0 for i in relevent_index])
        tf_r = 0 if term_occurence_count==0 else 1+log(term_occurence_count+document_occurence_count)
        df_c = max(sum([1 if t in verb_sets[i] else 0 for i in irrelevent_index]),0.5)
        idf = log(len(docs)  / df_c)
        tf_idf = tf_r*idf
        results.append((t,tf_idf, tf_r, document_occurence_count,idf,df_c))
    return results

def get_ngrams(data):
    # with open(data_path, 'r', encoding='utf-8') as file:
    #     data = file.readlines()

    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in data]
    unigrams = [word for sentence in tokenized_sentences for word in sentence]
    bigrams = [gram for sentence in tokenized_sentences for gram in ngrams(sentence, 2, pad_left=True, pad_right=True)]
    unigrams = FreqDist(unigrams)
    bigrams = FreqDist(bigrams)    
    return unigrams, bigrams


def reorder_query(unigram, bigram, query):

    original_query = query.split()
    new_added_words = original_query[-2:]
    original_query = original_query[:-2]

    query_order_permutations = []
    for i in range(len(original_query)+1):
        for j in range(len(original_query)+1):
            if i < j or i == j:
                new_list = deepcopy(original_query)
                new_list.insert(j, new_added_words[1])
                new_list.insert(i, new_added_words[0])
                query_order_permutations.append(new_list)
            if j < i or j == i:
                new_list = deepcopy(original_query)
                new_list.insert(i, new_added_words[0])
                new_list.insert(j, new_added_words[1])
                query_order_permutations.append(new_list)
    best_query_order = query_order_permutations[0]
    max_score = 0
    for query_order in query_order_permutations:
        score = sum(log2(1+bigrams[(query_order[i], query_order[i+1])]/unigram[query_order[i]]) if unigram[query_order[i]] != 0 else 0 for i in range(len(query_order)-1))
        if score > max_score:
            max_score = score
            best_query_order = query_order
    reordered_query = ' '.join(best_query_order)
    # print(f'reordered query: {reordered_query}')
    return reordered_query

if __name__ == "__main__":
    #  Read parameters
    API_key = sys.argv[1]
    search_engine_id = sys.argv[2]
    target_precision = float(sys.argv[3])
    query = ' '.join(sys.argv[4:])
    query = query.strip('"')

    # Read stopwords
    stopwords_set = set()
    with open('proj1-stop.txt', 'r') as file:
        for line in file:
            stopwords_set.add(line.strip())

    # get ngrams
    try:
        nltk.data.find('tokenizers/punkt')
        # print('have already downloaded nltk punkt')
    except LookupError:
        print('download nltk punkt')
        nltk.download('punkt')

    # start
    relevant_docs = []
    now_precision = 0.0
    while now_precision < target_precision:
        # get 10 results from google
        print(f'Parameters:\nClient key  = {API_key}\nEngine key  = {search_engine_id}\nQuery       = {query}\nPrecision   = {target_precision}\nGoogle Search Results:\n{"=" * 22}', end='')
        num_related_doc = 0
        num_total_doc = 0
        docs = getResultFromGoogle(query, API_key, search_engine_id)
        relevant_index = []
        for (i, doc) in enumerate(docs, 1):
            print(f'\nResult {i}')
            print(f'URL: {doc["URL"]}')
            print(f'Title: {doc["Title"]}')
            print(f'Summary: {doc["Summary"]}')
            # if 'fileFormat' in doc:
            #     print(f'!!! non-html file (fileFormat: {doc["fileFormat"]}), IGNORE !!!')
            #     continue

            num_total_doc += 1
            relevant = input('\nRelevant (Y/N)? ').strip().lower()
            if relevant == 'y':
                num_related_doc += 1
                relevant_index.append(i-1)
                relevant_docs.append(doc["Summary"])
                relevant_docs.append(doc["Title"])
        
        now_precision = num_related_doc / num_total_doc

        # no relevant docs in this iteration, terminate
        print(f'{"=" * 22}\nFEEDBACK SUMMARY\nQuery {query}\nPrecision {now_precision}')
        if now_precision == 0:
            print(f'No relevant documents found in this iteration, terminate')
            break
        # still below the desired precision
        elif now_precision < target_precision:
            print(f'Still below the desired precision of {target_precision}')
            print('Indexing results ....')
            # calculate scores for all candidate words
            term_scores = sorted(calculate_scores(docs, relevant_index, stopwords_set, query), key=lambda x: x[1], reverse=True)
            # for i in range(10):
            #     print(term_scores[i])
            # if there are too many words that could be the candidate of the second word, choose to include one 1 new words
            if term_scores[1][1] == term_scores[5][1]:
                query = f'{query} {term_scores[0][0]}'
                print(f'Augmenting by  {term_scores[0][0]}')
            else:
                query = f'{query} {term_scores[0][0]} {term_scores[1][0]}'
                print(f'Augmenting by  {term_scores[0][0]} {term_scores[1][0]}')
            # construct ngram model based on all relevant docs and reorder the augmented query
            unigrams, bigrams = get_ngrams(relevant_docs)
            query = reorder_query(unigrams, bigrams, query)
            print(f'New query after reordering: {query}')
        # reach the desired precision
        else:
            print(f'Desired precision reached, done')
            break
