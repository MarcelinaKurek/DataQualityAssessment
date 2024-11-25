import numpy as np
import os
import pandas as pd
import re
import requests
import torch
import warnings

from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

class LanguageWarning(Warning):
    pass

class LanguageValidator:
    def __init__(self, data):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.data = np.array(data.reset_index())
        self.vocab = self.tokenizer.get_vocab()
        self.abbr_df = self.load_abbreviations()
        self.result_df = None

    # @staticmethod
    # def print_dict(result_dict):
    #     for key, value in result_dict.items():
    #         print(f"{key}: {'; '.join(value)}")

    def regex_check(self, values_list, pat):
        """
        Function to check if any names match a regex pattern
        :param categories_df: dataframe containing categories
        :return: dictionary with potentially suspicious names
        """
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

    def check_oov(self):
        """
        Check if any word is not present in the BERT vocabulary
        :param values_list: list of distinct values from a column
        :return: list of words not present in the vocabulary
        """
        oov_df = [] #result dataframe
        oov_words = [] #list of words not found in standard vocabulary
        oov_tokens =[] #context of words not found in vocabulary
        iv_words =[] #words present in vocabulary
        lemma_tokens = ['##s', '##ed', '##ing']
        for i in range(len(self.data)):
            values_list = self.data[i][1]
            for entry in values_list:
                entry = re.sub(r'[^a-zA-Z0-9]', ' ', entry)
                tokenized = self.tokenizer.tokenize(entry)
                #find tokens which start with ## as they indicate part of unknown word
                unknown_tokens = list(map(lambda x: x.startswith('##') and x not in lemma_tokens, tokenized))
                unknown_token_idx = list(np.where(unknown_tokens)[0])
                if any(unknown_tokens):
                    for i in unknown_token_idx:
                        if i+1 not in unknown_token_idx:
                            word = ''.join([tokenized[i-1], tokenized[i].replace('##', '')])
                        else:
                            word = ''.join([tokenized[i - 1], tokenized[i].replace('##', ''), tokenized[i+1].replace('##', '')])
                        if not word.startswith("##") and word not in oov_words:
                            oov_tokens.append(entry)
                            oov_words.append(word)
                elif tokenized not in iv_words:
                    tokenized = ' '.join(tokenized).replace(" ##", "")
                    iv_words.append(tokenized)
            oov_df.append([iv_words, oov_words, oov_tokens])
            oov_words = []
            iv_words = []
            oov_tokens = []
        return pd.DataFrame(oov_df, columns=['words', 'oov_words', 'oov_tokens'], index=self.data[:,0])

    def invalid_characters_check(self):
        errors_df = []
        for i in range(len(self.data)):
            values_list = self.data[i][1]
            potential_errors = self.regex_check(values_list, pat=r"$#*?:;!")
            errors_df.append(potential_errors)
        return pd.DataFrame(errors_df, index=self.data[:,0], columns=['Invalid Values']) if errors_df != [[]] else None

    def abbreviations_check(self, values_df):
        abbreviations_df = []
        found_abbreviations = []
        not_abbreviations = []
        not_abbreviations_full_tokens = []
        abbr_list = self.abbr_df['abbreviation'].tolist()
        abbr_list = list(map(lambda x: x.lower().replace('.', '').split(' ')[0], abbr_list))
        for i in range(len(values_df)):
            values_list = values_df.iloc[i, 1]
            for j in range(len(values_list)):
                if values_list[j] not in abbr_list:
                    not_abbreviations.append(values_list[j])
                    not_abbreviations_full_tokens.append(values_df.iloc[i][2][j])
                else:
                    found_abbreviations.append(values_list[j])
            abbreviations_df.append([found_abbreviations if found_abbreviations != [] else None,
                                     not_abbreviations if not_abbreviations != [] else None,
                                     not_abbreviations_full_tokens if not_abbreviations_full_tokens != [] else None])
            found_abbreviations, not_abbreviations, not_abbreviations_full_tokens = [], [], []
        return pd.DataFrame(abbreviations_df, index=self.data[:,0], columns=['Abbreviation', 'Out of Vocabulary', 'OOV Full Entries'])

    def load_abbreviations(self):
        if not os.path.exists("abbreviations.txt"):
            try:
                r = requests.get(
                    'https://raw.githubusercontent.com/ePADD/muse/refs/heads/master/WebContent/WEB-INF/classes/dictionaries/en-abbreviations.txt')
                text = r.text
                with open("abbreviations.txt", "w") as file:
                    file.write(text)
            except:
                raise Exception("Could not download abbreviation file")
        df = pd.read_csv("abbreviations.txt", sep=r"\s{2,}", comment='#', engine='python')
        df.columns = ['abbreviation', 'word']
        print("Abbreviations successfully loaded")
        return df

    def run_language_validator(self):
        """
        Function to perform language validation.
        :return:
        """
        print("\nRunning language validator...")
        invalid_characters = self.invalid_characters_check()
        oov_df = self.check_oov()
        abbreviations_df = self.abbreviations_check(oov_df)

        if invalid_characters is not None:
            invalid_dict = invalid_characters.dropna().to_dict()['Invalid Values']
            warnings.warn("Invalid values found in columns: "+", ".join(f"{key}: {value}" for key, value in invalid_dict.items()),
                          category=LanguageWarning, stacklevel=5)
        if list(abbreviations_df.loc[:,'Abbreviation']) != [[]]:
            abb_df = abbreviations_df.loc[:,'Abbreviation'].dropna()
            warnings.warn(f"Out of vocabulary words that might be abbreviations:\n{abb_df}",
                          category=LanguageWarning, stacklevel=5)
        if list(abbreviations_df['Out of Vocabulary']) != [[]]:
            oov_df = abbreviations_df['Out of Vocabulary'].dropna()
            warnings.warn(f"Out of vocabulary words that might be invalid:\n{oov_df}",
                          category=LanguageWarning, stacklevel=5)
        print("Language validator successfully completed")

        # abbreviations_df['Invalid Values'] = invalid_characters['Invalid Values']
        # abbreviations_df = abbreviations_df.drop(columns=['Abbreviation'], axis=1)
        # new_order = ['Invalid Values', 'Out of Vocabulary', 'OOV Full Entries']
        # abbreviations_df = abbreviations_df[new_order]
        # abbreviations_df['OOV Full Entries'] = abbreviations_df['OOV Full Entries'].apply(lambda x: '; '.join(x))
        # abbreviations_df['Out of Vocabulary'] = abbreviations_df['Out of Vocabulary'].apply(lambda x: '; '.join(x))
        # print(abbreviations_df.reset_index().to_latex(index=False, na_rep='', label="Adult_result_table", caption='Abbreviations'))
        # categories_embeddings = self.generate_embeddings(self.data[:,0])
        # cnt = 0
        # for i in self.data[:,0]:
        #     values_list = oov_df.loc[i, 'words']
        #     if values_list != []:
        #         values_embeddings = self.generate_embeddings(oov_df.loc[i, 'words'])
        #         cosine_similarity_category_name = cosine_similarity([categories_embeddings[cnt]], values_embeddings)[0]
        #         not_similar_to_category_name = np.where(cosine_similarity_category_name < 0.75)[0]
        #         if len(not_similar_to_category_name) != 0:
        #             print(f"Words potentially not matching a category {i}:", end=' ')
        #             print(np.array(values_list)[not_similar_to_category_name])
        #     cnt += 1





