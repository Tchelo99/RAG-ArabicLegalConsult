# RAG Application for Arabic Legal Documents

This project is a Retrieval-Augmented Generation (RAG) application built using Python and the LangChain framework. It is designed to query Arabic legal documents and provide accurate, context-based answers.

## Project Overview

The RAG app leverages the following components:

- **LangChain**: A framework for building applications powered by language models.
- **Hugging Face Transformers**: Used for text embeddings.
- **Chroma**: For storing and retrieving document embeddings.

## PDF Documents Used

This application processes and retrieves information from the following legal documents:

- **قانون العمل الجزائري رقم 90-11 لسنة 1990**
- **قانون رقم 23-12 يحدد القواعد العامة المتعلقة بالصفقات العمومية**

## Features

- **Arabic Language Support**: The app supports Arabic queries and provides responses in Arabic.
- **Efficient Document Retrieval**: Utilizes Chroma for fast and accurate document retrieval based on text embeddings.
- **Custom Prompting**: The app includes custom prompts in Arabic to better suit the context of legal queries.
