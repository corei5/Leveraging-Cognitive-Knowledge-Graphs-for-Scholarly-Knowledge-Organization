import pandas as pd
from .data_preprocessing import keep_one_keyword

def create_df_with_prompts_research_field(data, clean_abstract, research_field_data):
    # Create a list of system prompts (same for all rows)
    # data = data.dropna(subset=['doi'])
    data['unpaywall_abstract'] = data['unpaywall_abstract'].apply(clean_abstract)
    system_prompt = ["Answer the following question."] * len(data)

    # Create user prompts and model outputs based on the provided data
    # user_prompts = [
    #     f"Your task is to recommend the research field depending on the title and abstract. The "
    #     f"research field must be recommended from this list: '{research_field_data}'. This is a "
    #     f"title:'{paper_title}' and abstract: '{unpaywall_abstract}' of a research paper."
    #     for paper_title, unpaywall_abstract in zip(data['paper_title'], data['unpaywall_abstract'])
    # ]
    user_prompts = [
        (f"This is the title: {paper_title} and abstract: {unpaywall_abstract}' of a research paper."
         f" What will be the research field for this research paper?"
         f" The research field must recommended from this list of research fields: {research_field_data}.")
        for paper_title, unpaywall_abstract in zip(data['paper_title'], data['unpaywall_abstract'])
    ]
    model_output = data['research_field_label']

    # Create a DataFrame with the specified columns
    df = pd.DataFrame({'system_prompt': system_prompt, 'user_prompt': user_prompts, 'model_output': model_output})

    return df

def create_df_with_prompts_predicate_label(data, clean_abstract):
    data['predicateLabel'] = data['predicateLabel'].apply(keep_one_keyword)
    data['unpaywall_abstract'] = data['unpaywall_abstract'].apply(clean_abstract)

    # Create a list of system prompts (same for all rows)
    system_prompt = ["Answer the following question."] * len(data)

    # Create user prompts and model outputs based on the provided data

    # user_prompts = [
    #     f"Your task is to recommend the list of predicates depending on the title and abstract. "
    #     f"This is the title: {paper_title} and abstract: {unpaywall_abstract}' of a research paper. An example list of predicates such as method, research problem, dataset, results and many more that can be used to describe this research paper based on the given title and abstract."
    #     for paper_title, unpaywall_abstract in zip(data['paper_title'], data['unpaywall_abstract'])
    # ]
    user_prompts = [
        (f"This is the title: {paper_title} and abstract: {unpaywall_abstract}' of a research paper."
         f" What will be the list of predicates, such as method, research problem, dataset, results, and many more, "
         f"that can be used to comprehensively describe this research paper?")
        for paper_title, unpaywall_abstract in zip(data['paper_title'], data['unpaywall_abstract'])
    ]
    model_output = data['predicateLabel']

    # Create a DataFrame with the specified columns
    df = pd.DataFrame({'system_prompt': system_prompt, 'user_prompt': user_prompts, 'model_output': model_output})
    df.to_csv("Dataset/clean_df_predicates.csv", index=False)

    return df

def create_df_with_prompts_object_label(data, clean_abstract):
    data['unpaywall_abstract'] = data['unpaywall_abstract'].apply(clean_abstract)
    #data['predicateLabel'] = data['predicateLabel'].apply(keep_one_keyword)
    # Create a list of system prompts (same for all rows)
    system_prompt = ["Answer the following question."] * len(data)

    # Create user prompts and model outputs based on the provided data

    # user_prompts = [
    #     f"You have to do a QA task from the title and abstract. This is the title:'{paper_title}' and abstract: '{unpaywall_abstract}' of a research paper. "
    #     f"What is the '{predicateLabel}' described in this research paper? Write it in less than 5 words. "
    #     for paper_title, unpaywall_abstract, predicateLabel in zip(data['paper_title'], data['unpaywall_abstract'], data['predicateLabel'])
    # ]
    user_prompts = [
        (f"This is the title: '{paper_title}' and abstract: '{unpaywall_abstract}' of a research paper."
         f" What is the '{predicateLabel}' described in this research paper? Please provide a concise description in less than 5 words.")
        for paper_title, unpaywall_abstract, predicateLabel in
        zip(data['paper_title'], data['unpaywall_abstract'], data['predicateLabel'])
    ]
    model_output = data['concatenatedObjectLabel']

    # Create a DataFrame with the specified columns
    df = pd.DataFrame({'system_prompt': system_prompt, 'user_prompt': user_prompts, 'model_output': model_output})

    return df