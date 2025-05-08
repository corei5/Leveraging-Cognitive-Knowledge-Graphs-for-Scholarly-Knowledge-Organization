import re

def keep_one_keyword(row):
    words = row.split(',')
    keyword_counts = {}
    new_row = []
    for word in words:
        word = word.strip()
        if word not in keyword_counts:
            keyword_counts[word] = 1
            new_row.append(word)
        elif keyword_counts[word] == 1:
            keyword_counts[word] += 1
            new_row.remove(word)
            new_row.append(word)
    #print(new_row)
    return ', '.join(new_row)

def clean_abstract(text):
    # Remove extra whitespaces, leading/trailing spaces, and special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'<.*?>', '', text)
    #text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    html_regex = '<.*?>'
    return ' '.join(re.sub(html_regex, ' ', text).split()).lower()
    #
    # return text