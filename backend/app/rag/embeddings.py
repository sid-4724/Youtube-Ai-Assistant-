from langchain_huggingface import HuggingFaceEmbeddings


def load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Load the configured local sentence-transformer on CPU by default."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
