import pandas as pd

#Example for the Test data

df_research_filed = pd.read_csv('Dataset/task_driven/test/prompt_df_research_filed_test.csv')
df_list_of_predicates = pd.read_csv('Dataset/task_driven/test/prompt_df_predicate_label_test.csv')
df_objects = pd.read_csv('Dataset/task_driven/test/prompt_df_object_label_test.csv')

df_research_filed['user_prompt'] = "Research field prediction: " + df_research_filed['user_prompt']
df_list_of_predicates['user_prompt'] = "List of predicates prediction: " + df_list_of_predicates['user_prompt']
df_objects['user_prompt'] = "Objects prediction: " + df_objects['user_prompt']

df_research_filed.to_csv("Dataset/task_driven/test/prefix/prefix_prompt_df_research_filed_test.csv", index=False)
df_list_of_predicates.to_csv("Dataset/task_driven/test/prefix/prefix_prompt_df_predicate_label_test.csv", index=False)
df_objects.to_csv("Dataset/task_driven/test/prefix/prefix_prompt_df_object_label_test.csv", index=False)
