import numpy as np
from urllib.parse import urlparse
import urllib.parse
import requests
from .data_preprocessing import clean_abstract
orkg_mail = 'info@orkg.org'
def extract_abstract(df):
    df = df.dropna(subset=['doi'])
    df = df.groupby('doi').agg({'predicateLabel': ', '.join, 'paper_title': 'first', 'contribution': 'first',
                                  'research_field_label': 'first'}).reset_index()
    df['unpaywall_abstract'] = np.nan
    row, col = df.shape
    for i in range(0, row, 1):
        print(i)
        doi = df['doi'][i]
        url_encoded_doi = urllib.parse.quote_plus(doi.replace(" ", ""))
        url = 'https://api.crossref.org/works/{}'.format(url_encoded_doi)
        response = requests.get(url)
        try:
            data = response.json()
            abstract = data['message'].get('abstract', '').replace("<jats:title>Abstract</jats:title><jats:p>", "")
            abstract = abstract.replace("<jats:p>", "")
            abstract = abstract.replace("</jats:p>", "")
            abstract = abstract.replace("<jats:title>Abstract</jats:title><jats:sec><jats:label />", "")
            abstract = abstract.replace(".</jats:p></jats:sec>", "")
            df['unpaywall_abstract'][i] = clean_abstract(abstract)
        except:
            pass

    df.to_csv("queryv2_unpaywall.csv")
    return df
