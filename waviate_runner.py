from modules import *
from weaviate_4 import WeaviateFieldClusterer

embedding_dict = {
    "FirstNameEmbeddings_1": "fname",
    "LastNameEmbeddings_1": "lname",
    "StreetNumberEmbeddings": "stNo",
    "AddressLine1Embeddings": "add1",
    "AddressLine2Embeddings": "add2",
    "CityEmbeddings_1": "city",
    "AreaCodeEmbeddings": "areacode",
    "StateEmbeddings": "state",
    "DOBEmbeddings": "dob",
    "SSNEmbeddings": "ssn"
}

thresholds = {
    "FirstNameEmbeddings_1": 0.15,
    "LastNameEmbeddings_1": 0.15,
    "StreetNumberEmbeddings": 0.1,
    "AddressLine1Embeddings": 0.2,
    "AddressLine2Embeddings": 0.2,
    "CityEmbeddings_1": 0.1,
    "AreaCodeEmbeddings": 0.05,
    "StateEmbeddings": 0.05,
    "DOBEmbeddings": 0.05,
    "SSNEmbeddings": 0.01
}

auth = Auth.api_key(api_key_weaviate)

clusterer = WeaviateFieldClusterer(
    cluster_url_weaviate=cluster_url_weaviate,
    api_key_weaviate=auth,
    model_name="all-MiniLM-L6-v2",
    input_file_name = "C:/Users/akshay.kudale/OneDrive - TIGER ANALYTICS INDIA CONSULTING PRIVATE LIMITED/GenAI/GenAI-MDM/data/sample_input_4.csv",
    output_file_name = "C:/Users/akshay.kudale/OneDrive - TIGER ANALYTICS INDIA CONSULTING PRIVATE LIMITED/GenAI/GenAI-MDM/data/sample_output_4.csv",
    embedding_dict=embedding_dict,
    thresholds=thresholds,
    api_key_llm=api_key_llm,
    base_url_llm=base_url_llm
)

clusterer.create_store_embeddings()
grouped_df = clusterer.group_similar_records_llm()
# print(grouped_df)
clusterer.close_connection()


