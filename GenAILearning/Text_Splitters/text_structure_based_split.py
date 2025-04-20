from langchain.text_splitter import RecursiveCharacterTextSplitter

text='''Machine Learning (ML) is a subset of artificial intelligence that focuses on building systems capable of learning from data and improving their performance over time without being explicitly programmed. These systems use algorithms to analyze data, identify patterns, and make predictions or decisions based on that information. Common types of machine learning include supervised learning, where models are trained on labeled data; unsupervised learning, which deals with finding structure in unlabeled data; and reinforcement learning, where models learn by interacting with an environment to maximize a reward.

ML has become a powerful tool across a wide range of industries. In healthcare, it helps in diagnosing diseases and personalizing treatment plans. In finance, it's used for fraud detection and algorithmic trading. In everyday applications, machine learning powers recommendation engines, voice assistants, and self-driving cars. As the availability of data and computational power grows, machine learning continues to evolve and shape the future of technology and innovation.'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

response = splitter.split_text(text)

print(response[0])