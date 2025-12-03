from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    chunks = []
    
    if language == "zh":
        separators = [
            "\n\n",
            "\n",
            "。",
            "！",
            "；",
            "，",
            "、",
            " ",
            ""
        ]
    else:
        separators = [
            "\n\n",
            "\n", 
            ".",
            " ",
            ""
        ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    for doc in docs:
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            text = doc['content']
            lang = doc['language']
            
            if lang == language:
                
                raw_text_chunks = text_splitter.split_text(text)
                
                doc_metadata = doc.copy()
                doc_metadata.pop('content', None)
                
                for chunk_count, content in enumerate(raw_text_chunks):
                    
                    chunk_metadata = doc_metadata.copy()
                    chunk_metadata['chunk_index'] = chunk_count
                    
                    chunk = {
                        'page_content': content.strip(),
                        'metadata': chunk_metadata
                    }
                    chunks.append(chunk)
    return chunks
