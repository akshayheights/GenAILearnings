import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, Configure, DataType
from sentence_transformers import SentenceTransformer
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

api_key_weaviate = 'SGpoRWYzZHVFOTdLMUdveF8zTlQ2QS9tam1UQTBIZElISHBETGtOUlY0MEtrNTFyeEhCaEdrYlVkZUJZPV92MjAw'
cluster_url_weaviate = 'mfp7py4mr1gvyrh13ptbgq.c0.asia-southeast1.gcp.weaviate.cloud'

base_url_llm="https://openrouter.ai/api/v1"
# api_key_llm="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"
api_key_llm="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"