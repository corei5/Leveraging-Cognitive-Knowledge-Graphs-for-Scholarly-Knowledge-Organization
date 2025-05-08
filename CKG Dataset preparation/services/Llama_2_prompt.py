def generate_prompt_Llama_2(row):
    system_prompt = row['system_prompt']
    user_prompt = row['user_prompt']
    model_output = row['model_output']

    formatted_text = f"<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_prompt} [/INST] {model_output} </s>"
    # formatted_text = f"<s>[INST] <<SYS>>You are an assistant for building a knowledge graph for science. Your task is to recommend the research field depending on the title and abstract. The research field must recommended from the '{research_field_data}' JSON. <</SYS>> This is a title:'{title}' and abstract: '{abstract}' of a research paper. [/INST] "
    # formatted_text += f"{research_field_label} </s>"
    return formatted_text