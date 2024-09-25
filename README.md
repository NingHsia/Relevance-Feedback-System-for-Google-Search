# Relevance Feedback System for Google Search
This project implements a query reformulation system that improves Google Search results by leveraging user-provided relevance feedback. It dynamically refines queries based on the user's input, aiming to increase the relevance of search results using a combination of a **modified TF-IDF algorithm** and **bigram query reordering**.

The system receives a user query and a target precision value for the top-10 search results (precision@10). The search results are retrieved via the Google Custom Search API, and users are asked to mark relevant pages. Based on this feedback, the query is automatically modified in each iteration until the desired precision@10 is met, or further improvement becomes impossible.

## Features
- **Revised TF-IDF Approach**: The system introduces a tailored TF-IDF algorithm for Google Search by prioritizing terms that consistently appear in user-identified relevant results and penalizing those that appear only in irrelevant ones. This allows for a more refined selection of terms for query reformulation.
- **Bigram Model for Query Reordering**: In addition to refining term selection, the system utilizes a bigram model to reorder query terms based on patterns observed in relevant search results. This improves the natural flow and structure of the reformulated queries, leading to more accurate results.
- 
## Prerequisite
- Google Custom Search API Key
- Google Search engine ID
- Python 3.x

## Installation
1. Clone the repository:
```bash
git clone https://github.com/NingHsia/Relevance-Feedback-System-for-Google-Search.git
```
2. Install required dependencies:
```bash
sudo apt-get -y update
sudo apt-get install python3-pip
sudo apt install python3-testresources
pip3 install --upgrade google-api-python-client
pip3 install nltk
```

## Usage
Run the query reformulation system:
```bash
python3 main.py <google api key> <google engine id> <precision> <query>
```
where:
- <google api key> is the Google Custom Search JSON API Key
- <google engine id> is the Google Custom Search Engine ID
- <precision> is the target value for precision@10, a real number between 0 and 1
- <query> is the query, a list of words in double quotes (e.g., “wojcicki”)
    
For example,
```bash
python3 main.py XXXXX XXXXX 0.9 "wojcicki"
```
