import itertools
import numpy as np
import os
import pandas as pd
import re
import requests
import torch
import warnings

from transformers import BertTokenizer, BertModel

from utils import make_necessary_folders

class LanguageWarning(Warning):
    pass

class LanguageValidator:
    def __init__(self, data, data_path, visualizer, linter_count, linters_dict):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.data = np.array(data['Names'].reset_index())
        self.categories_count = data['Categories nr']
        self.vocab = self.tokenizer.get_vocab()
        self.abbr_df = self.load_abbreviations()
        self.result_df = None
        self.nr_categories = np.array(data['Categories nr'])
        self.result_path = make_necessary_folders(data_path)
        self.visualizer = visualizer
        self.linter_count = linter_count
        self.linters_dict = linters_dict


    def regex_check(self, values_list, pat):
        """
        Function to check if any names match a regex pattern
        :param categories_df: dataframe containing categories
        :return: dictionary with potentially suspicious names
        """
        special_characters = []
        for entry in values_list:
            if re.search(pat, entry):
                special_characters.append(re.findall(pat, entry))
        return list(itertools.chain(*special_characters))

    # def generate_embeddings(self, values_list):
    #     """
    #     Function to generate embeddings using BERT model
    #     :param values_list: list of distinct values from a column
    #     :return: dictionary with embeddings
    #     """
    #     embeddings = []
    #     for entry in values_list:
    #         entry = entry.replace('-', ' ').replace('_', ' ')
    #         tokens = self.tokenizer(entry, return_tensors='pt')
    #         with torch.no_grad():
    #             output = self.model(**tokens)
    #             embedding = output.last_hidden_state[:, 0, :]  # CLS token embedding
    #         embeddings.append(embedding.squeeze().numpy())
    #     embeddings = np.array(embeddings)
    #     embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    #     return embeddings

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
                entry = re.sub(r'[^a-zA-Z]', ' ', entry) #letters only
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

    def special_characters_check(self):
        errors_df = []
        for i in range(len(self.data)):
            values_list = self.data[i][1]
            potential_errors = self.regex_check(values_list, pat=r"[^a-zA-Z\d+\s_-]")
            nums = self.regex_check(values_list, pat=r"\d+")
            errors_df.append([np.unique(potential_errors) if potential_errors != [] else None,
                              np.sort(np.unique(nums).astype(int)).astype(str) if nums != [] else None])
        return pd.DataFrame(errors_df, index=self.data[:,0], columns=['Special Characters', 'Numbers'])

    def abbreviations_check(self, values_df):
        abbreviations_df = []
        found_abbreviations = []
        not_abbreviations = []
        not_abbreviations_full_tokens = []
        full_words = []
        abbr_list = self.abbr_df['abbreviation'].tolist()
        full_list = self.abbr_df['word'].tolist()
        abbr_list = list(map(lambda x: x.lower().replace('.', '').split(' ')[0], abbr_list))
        for i in range(len(values_df)):
            values_list = values_df.iloc[i, 1]
            for j in range(len(values_list)):
                if values_list[j] not in abbr_list:
                    not_abbreviations.append(values_list[j])
                    not_abbreviations_full_tokens.append(values_df.iloc[i,2][j])
                else:
                    found_abbreviations.append(values_list[j])
                    idx = abbr_list.index(values_list[j])
                    full_words.append(full_list[idx])
            abbreviations_df.append([found_abbreviations if found_abbreviations != [] else None,
                                     full_words if full_words != [] else None,
                                     not_abbreviations if not_abbreviations != [] else None,
                                     not_abbreviations_full_tokens if not_abbreviations_full_tokens != [] else None])
            found_abbreviations, not_abbreviations, not_abbreviations_full_tokens, full_words = [], [], [], []
        return pd.DataFrame(abbreviations_df, index=self.data[:,0], columns=['Abbreviation', 'Full Word', 'Out of Vocabulary', 'OOV Full Entries'])

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

    def run_language_validator(self, save_summary=True):
        """
        Function to perform language validation.
        :return:
        """
        print("\nRunning language validator...")
        invalid_characters = self.special_characters_check()
        oov_df = self.check_oov()
        abbreviations_df = self.abbreviations_check(oov_df)
        merged_df = pd.merge(invalid_characters, abbreviations_df, left_index=True, right_index=True)
        all_none = merged_df.isna().all(axis=1)
        affected_columns = set(self.data[~all_none, 0])
        columns_max_10_categories = set(self.categories_count.loc[self.categories_count <= 10].index)
        columns_10_20_categories = set(self.categories_count.loc[(self.categories_count > 10) & (self.categories_count <= 20)].index)
        affected_max_10_categories = list(affected_columns & columns_max_10_categories)
        affected_10_20_categories = list(affected_columns & columns_10_20_categories)
        self.visualizer.plot_categories_count(affected_max_10_categories)
        self.visualizer.plot_categories_count(affected_10_20_categories, horizontal=True)

        if save_summary:
            merged_df.to_csv(f"{self.result_path}/language_summary_df.csv")
            print(f"Summary table saved to {self.result_path}/language_summary_df.csv")
        self.print_warnings(abbreviations_df, invalid_characters)
        print("\nLanguage validator successfully completed")

    def print_warnings(self, abbreviations_df, invalid_characters):
        self.print_invalid_characters_warnings(invalid_characters, 'Special Characters', 'Special characters detected', "lang_special_chars")
        self.print_invalid_characters_warnings(invalid_characters, 'Numbers', 'Numbers detected', "lang_numeric_vals")

        if any(abbreviations_df.loc[:,'Abbreviation']):
            abb_df = abbreviations_df.loc[:,['Abbreviation', 'Full Word']].dropna()
            warnings.warn(f"Out of vocabulary words that might be abbreviations detected",
                          category=LanguageWarning, stacklevel=7)
            self.linter_count[self.linters_dict["lang_abbr"]] += len(abb_df)
            print("\nDetected abbreviations out of vocabulary:")
            for x, y, z in zip(abb_df.index, abb_df['Abbreviation'], abb_df['Full Word']):
                print(f"{x}: {', '.join(y)} ({', '.join(z)})")

        if list(abbreviations_df['Out of Vocabulary']) != [[]]:
            oov_list_len = np.array(abbreviations_df['Out of Vocabulary'].apply(lambda x: len(x) if x is not None else 0.001))
            oov_df = abbreviations_df['Out of Vocabulary'].dropna()
            custom_values_idx = np.where((oov_list_len / self.nr_categories > 0.50))[0]
            if len(custom_values_idx) != 0:
                custom_columns = self.data[:,0][custom_values_idx]
                warnings.warn(f"Columns with high percentage of out of vocabulary values detected: {', '.join(custom_columns)}", category=LanguageWarning, stacklevel=5)
                self.linter_count[self.linters_dict["lang_oov_custom"]] += len(custom_columns)
            if len(oov_df) - len(custom_values_idx) != 0:
                warnings.warn(f"Out of vocabulary words that might be invalid detected",
                              category=LanguageWarning, stacklevel=6)
                self.linter_count[self.linters_dict["lang_oov_invalid"]] += len(oov_df) - len(custom_values_idx)
                print("\nPotentially invalid values:")
                for x, y in zip(oov_df.index, oov_df):
                    print(f"{x}: {', '.join(y)}")

    def print_invalid_characters_warnings(self, invalid_characters, col, mess, dict_label):
        if len(invalid_characters.loc[:,col].dropna()) != 0:
            invalid_df = invalid_characters[col].dropna()
            warnings.warn(f"{mess}", category=LanguageWarning, stacklevel=7)
            self.linter_count[self.linters_dict[dict_label]] += len(invalid_df)
            print(f"\n{mess}:")
            for x, y in zip(invalid_df.index, invalid_df):
                print(f"{x}: {', '.join(y)}")






