import time
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import urllib.parse
import requests
import re

from services.ORKG_research_field import research_field_data
from services.data_preprocessing import keep_one_keyword, clean_abstract
from services.paper_abstract import extract_abstract
from services.prompts_df import create_df_with_prompts_research_field, create_df_with_prompts_object_label, \
    create_df_with_prompts_predicate_label
from services.custom_test_and_train_split import custom_split_df
from services.Llama_2_prompt import generate_prompt_Llama_2
# ..................................................................
# doi, title, abstract, system_prompt, user_prompt, Llama_2_prompt, output (research field, predicate label), contribution
# ...................................................................
research_field_data_list = research_field_data()
research_field_data = ' '.join(research_field_data_list)



def orkg_object(df, df_abstract):
    df = df.dropna(subset=['doi'])
    df = df[df['concatenatedObjectLabel'] != 'non']
    df = df.reset_index(drop=True)
    df_abstract_obj = df.merge(df_abstract, on='doi', how='left')
    df_abstract_obj = df_abstract_obj.dropna(subset=['unpaywall_abstract'])
    df_abstract_obj = df_abstract_obj.reset_index(drop=True)
    df_abstract_obj['unpaywall_abstract'] = df_abstract_obj['unpaywall_abstract'].apply(clean_abstract)
    selected_columns = ['doi', 'paper_title_x', 'unpaywall_abstract', 'research_field_label_x', 'predicateLabel_x',
                        'concatenatedObjectLabel', 'contribution_x']
    new_df = df_abstract_obj[selected_columns]
    new_column_names = ['doi', 'paper_title', 'unpaywall_abstract', 'research_field_label', 'predicateLabel',
                        'concatenatedObjectLabel', 'contribution']
    new_df.columns = new_column_names
    return new_df


df = pd.read_csv('query.csv')
df_abstract = pd.read_csv('queryv3_abstract_unpaywall.csv')

# research filed related experiment

clean_df_for_split = orkg_object(df, df_abstract)
print(clean_df_for_split.columns.tolist())
clean_df_train,clean_df_test  = custom_split_df(clean_df_for_split)

#train data

prompt_df_research_filed_train = create_df_with_prompts_research_field(clean_df_train, clean_abstract, research_field_data)
prompt_df_research_filed_train = prompt_df_research_filed_train.drop_duplicates(subset=['user_prompt'])
print(prompt_df_research_filed_train)
prompt_df_research_filed_train.to_csv("Dataset/task_driven/train/prompt_df_research_filed_train.csv")

#test data

prompt_df_research_filed_test = create_df_with_prompts_research_field(clean_df_test, clean_abstract, research_field_data)
prompt_df_research_filed_test = prompt_df_research_filed_test.drop_duplicates(subset=['user_prompt'])
print(prompt_df_research_filed_test)
prompt_df_research_filed_test.to_csv("Dataset/task_driven/test/prompt_df_research_filed_test.csv")


# list of predicats related experiment

#This df is different

clean_df_train_predicate,clean_df_test_predicate  = custom_split_df(df_abstract)

#train data

prompt_df_predicate_label_train = create_df_with_prompts_predicate_label(clean_df_train_predicate, clean_abstract)
prompt_df_predicate_label_train = prompt_df_predicate_label_train.drop_duplicates(subset=['user_prompt'])
print(prompt_df_predicate_label_train)
prompt_df_predicate_label_train.to_csv("Dataset/task_driven/train/prompt_df_predicate_label_train.csv")

# test data

prompt_df_predicate_label_test = create_df_with_prompts_predicate_label(clean_df_test_predicate, clean_abstract)
prompt_df_predicate_label_test = prompt_df_predicate_label_test.drop_duplicates(subset=['user_prompt'])
print(prompt_df_predicate_label_test)
prompt_df_predicate_label_test.to_csv("Dataset/task_driven/test/prompt_df_predicate_label_test.csv")

#object related experiment

#train data

prompt_df_object_label_train = create_df_with_prompts_object_label(clean_df_train, clean_abstract)
prompt_df_object_label_train = prompt_df_object_label_train.drop_duplicates(subset=['user_prompt'])
print(prompt_df_object_label_train)
prompt_df_object_label_train.to_csv("Dataset/task_driven/train/prompt_df_object_label_train.csv")

# test data

prompt_df_object_label_test = create_df_with_prompts_object_label(clean_df_test, clean_abstract)
prompt_df_object_label_test = prompt_df_object_label_test.drop_duplicates(subset=['user_prompt'])
print(prompt_df_object_label_test)
prompt_df_object_label_test.to_csv("Dataset/task_driven/test/prompt_df_object_label_test.csv")

# merge train data

merge_train_df = pd.concat([prompt_df_research_filed_train, prompt_df_predicate_label_train, prompt_df_object_label_train], axis=0)
merge_train_df.to_csv("Dataset/task_independent/ORKG_LLM_Prompts_Train.csv", index=False)

merge_train_df['text'] = merge_train_df.apply(lambda row: f"{generate_prompt_Llama_2(row)}", axis=1)

#train

prompt_df_research_filed_train['text'] = prompt_df_research_filed_train.apply(lambda row: f"{generate_prompt_Llama_2(row)}", axis=1)
prompt_df_predicate_label_train['text'] = prompt_df_predicate_label_train.apply(lambda row: f"{generate_prompt_Llama_2(row)}", axis=1)
prompt_df_object_label_train['text'] = prompt_df_object_label_train.apply(lambda row: f"{generate_prompt_Llama_2(row)}", axis=1)

#test

prompt_df_research_filed_test['text'] = prompt_df_research_filed_test.apply(lambda row: f"{generate_prompt_Llama_2(row)}", axis=1)
prompt_df_predicate_label_test['text'] = prompt_df_predicate_label_test.apply(lambda row: f"{generate_prompt_Llama_2(row)}", axis=1)
prompt_df_object_label_test['text'] = prompt_df_object_label_test.apply(lambda row: f"{generate_prompt_Llama_2(row)}", axis=1)


print(merge_train_df)

merge_train_df[['text']].to_json("Dataset/task_independent/train_data.jsonl", orient='records', lines=True)

#train data save as Jsonl
prompt_df_research_filed_train[['text']].to_json("Dataset/task_driven/train/prompt_df_research_filed_train.jsonl", orient='records', lines=True)
prompt_df_predicate_label_train[['text']].to_json("Dataset/task_driven/train/prompt_df_predicate_label_train.jsonl", orient='records', lines=True)
prompt_df_object_label_train[['text']].to_json("Dataset/task_driven/train/prompt_df_object_label_train.jsonl", orient='records', lines=True)

#test data save as Jsonl
prompt_df_research_filed_test[['text']].to_json("Dataset/task_driven/test/prompt_df_research_filed_test.jsonl", orient='records', lines=True)
prompt_df_predicate_label_test[['text']].to_json("Dataset/task_driven/test/prompt_df_predicate_label_test.jsonl", orient='records', lines=True)
prompt_df_object_label_test[['text']].to_json("Dataset/task_driven/test/prompt_df_object_label_test.jsonl", orient='records', lines=True)
