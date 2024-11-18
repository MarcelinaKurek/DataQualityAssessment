from transformers import BertTokenizer, BertModel
from sklearn.ensemble import IsolationForest
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re


class LanguageValidator:
    def __init__(self, data):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.data = data
        self.vocab = self.tokenizer.get_vocab()

    def regex_check(self, values_list):
        """
        Function to check if any names match a regex pattern
        :param categories_df: dataframe containing categories
        :return: dictionary with potentially suspicious names
        """
        pat = r"$#*?:;!\."
        susp_values_list = []
        for entry in values_list:
            if re.findall(f'[{pat}]', entry):
                susp_values_list.append(entry)
        return susp_values_list

    def generate_embeddings(self, values_list):
        """
        Function to generate embeddings using BERT model
        :param values_list: list of distinct values from a column
        :return: dictionary with embeddings
        """
        embeddings = []
        for entry in values_list:
            entry = entry.replace('-', ' ').replace('_', ' ')
            tokens = self.tokenizer(entry, return_tensors='pt')

            with torch.no_grad():
                output = self.model(**tokens)
                embedding = output.last_hidden_state[:, 0, :]  # CLS token embedding
            embeddings.append(embedding.squeeze().numpy())
        embeddings = np.array(embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def check_oov(self, values_list):
        """
        Check if any word is not present in the BERT vocabulary
        :param values_list: list of distinct values from a column
        :return: list of words not present in the vocabulary
        """
        oov_words = []
        lemma_tokens = ['##s', '##ed', '##ing']
        for entry in values_list:
            entry = entry.replace('_', ' ').replace('-', ' ')
            tokenized = self.tokenizer.tokenize(entry)
            unknown_tokens = list(map(lambda x: x.startswith('##') and x not in lemma_tokens, tokenized))
            unknown_token_idx = list(np.where(unknown_tokens)[0])
            if any(unknown_tokens):
                for i in unknown_token_idx:
                    if i+1 not in unknown_token_idx:
                        word = ''.join([tokenized[i-1], tokenized[i].replace('##', '')])
                    else:
                        word = ''.join([tokenized[i - 1], tokenized[i].replace('##', ''), tokenized[i+1].replace('##', '')])
                    if not word.startswith("##"):
                        oov_words.append(word)
        return pd.unique(oov_words)

    def run_language_validator(self):
        """
        Function to perform language validation.
        :return:
        """
        df = np.array(self.data.reset_index())
        for i in range(len(df)):
            category_name = df[i][0]
            category_name = category_name.replace('_', ' ').replace('-', ' ')
            category_embedding =self.generate_embeddings([category_name])
            values_list = np.array(df[i][1])
            oov_words = self.check_oov(values_list)
            embeddings = self.generate_embeddings(values_list)
            cosine_similarity_category_name = cosine_similarity(category_embedding, embeddings)[0]
            not_similar_to_category_name = np.where(cosine_similarity_category_name < 0.75)
            if len(not_similar_to_category_name[0]) != 0:
                print(f"Words potentially not matching a category {category_name}:", end=' ')
                print(values_list[not_similar_to_category_name])
            if any(oov_words):
                print(f"Words not present in a standard dictionary for category {category_name}:", end=' ')
                print(oov_words)




