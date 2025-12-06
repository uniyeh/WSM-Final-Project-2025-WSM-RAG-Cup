from nltk.tokenize import sent_tokenize, word_tokenize
import re

def split_sentences(text, language):
    if language == 'zh':
        # Simple regex for Chinese sentence splitting
        return re.split(r'(?<=[。！？])', text)
    else:
        # Use NLTK for English
        # no need to remove these Ltd. Inc. etc.
        try:
            return sent_tokenize(text)
        except LookupError:
            import nltk
            nltk.download('punkt')
            nltk.download('punkt_tab')
            return sent_tokenize(text)

def chunk_row_chunks(docs, language):
    chunks = []
    for doc_index, doc in enumerate(docs):
        text = doc['page_content']
        sentences = split_sentences(text, language)
        chunk_count = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            chunk_metadata = doc.copy()
            chunk_metadata.pop('page_content', None)
            chunk_metadata['chunk_index'] = chunk_count
            chunks.append({
                'page_content': sentence,
                'metadata': chunk_metadata
            })
            chunk_count += 1
            
    return chunks

