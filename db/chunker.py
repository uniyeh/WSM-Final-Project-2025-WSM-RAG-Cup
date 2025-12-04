# here for chunking strategy
def single_chunk(content):
    chunks = []
    text = content
    text_len = len(text)
    start_index = 0
    while start_index < text_len:
        # chunk by \n
        end_index = text.find("\n", start_index)
        if end_index == -1:
            end_index = text_len
        chunk_metadata = {}
        # skip empty chunk
        if (text[start_index:end_index].strip() != ""):
            chunk = {
                'page_content': text[start_index:end_index],
                'metadata': chunk_metadata
            }
            chunks.append(chunk)
        start_index = end_index + 1
    return chunks
