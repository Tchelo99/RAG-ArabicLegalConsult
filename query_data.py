import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from get_embedding_function import get_embedding_function
from transformers import BertForMaskedLM, BertTokenizer
import torch


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
أجب عن السؤال بناءً على السياق التالي فقط:

{context}

---

أجب عن السؤال بناءً على السياق أعلاه: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Use an Arabic-compatible language model
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=2)

    # Prepare context and prompt
    max_context_tokens = 250  # Further reduced
    context_text = ""
    for doc, _score in results:
        new_context = doc.page_content + "\n\n---\n\n"
        if (
            len(tokenizer.encode(context_text + new_context + query_text))
            > max_context_tokens
        ):
            break
        context_text += new_context

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Ensure the entire prompt fits within the model's limit
    encoded_prompt = tokenizer.encode(
        prompt, truncation=True, max_length=510, return_tensors="pt"
    )

    # Generate response (using masked language modeling instead of generation)
    with torch.no_grad():
        output = model(encoded_prompt)
        predictions = output.logits.argmax(dim=-1)

    response_text = tokenizer.decode(predictions[0], skip_special_tokens=True)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
