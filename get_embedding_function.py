from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(
        model_name="aubmindlab/bert-base-arabertv02",
        model_kwargs={"trust_remote_code": True},
    )
    return embeddings
