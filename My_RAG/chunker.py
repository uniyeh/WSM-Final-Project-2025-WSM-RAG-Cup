def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for doc_index, doc in enumerate(docs):
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            text = doc['content']
            text_len = len(text)
            lang = doc['language']
            start_index = 0
            chunk_count = 0
            if lang == language:
                # 動態斷詞調整 chunk_size
                if text_len < 500:
                    effective_chunk_size = 300
                    effective_chunk_overlap = 50
                elif text_len < 2000:
                    effective_chunk_size = 200
                    effective_chunk_overlap = 50
                else:
                    effective_chunk_size = 150
                    effective_chunk_overlap = 30

                while start_index < text_len:
                    end_index = min(start_index + effective_chunk_size, text_len)
                    chunk_metadata = doc.copy()
                    chunk_metadata.pop('content', None)
                    chunk_metadata['chunk_index'] = chunk_count
                    chunk = {
                        'page_content': text[start_index:end_index],
                        'metadata': chunk_metadata
                    }
                    chunks.append(chunk)
                    start_index += effective_chunk_size - effective_chunk_overlap
                    chunk_count += 1
    return chunks
