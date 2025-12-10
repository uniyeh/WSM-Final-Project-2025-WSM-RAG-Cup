from retriever import create_retriever, get_chunks_from_db
from generator import generate_answer
from ollama import Client

def llm_router_chain(query, language):
    query_text = query['query']['content']
    
    # 1. Do the query expansion
    # new_query = expand_query(query_text, language)
    new_query = expand_query_2(query_text, language)
    print("new_query: ", new_query)
    # 2. Retrieve chunks
    retrieved_chunks = retrieve_chunks(new_query, language)
    # 3. Generate answer
    answer = generate_answer_llm(new_query, retrieved_chunks, language)
    # 4. Return answer and chunks
    return answer, retrieved_chunks

def expand_query(query, language="en", size=3):
    if language == "zh":
        prompt = f"""你是一位專業的搜尋優化專家。
        請遵照以下步驟為目標查詢的每個關鍵方面提供{size}個額外的資訊，使其更容易找到相關文檔。
        
        **步驟:**
        1. 提取目標查詢中的名詞，包含但不限於：年份、月份、地點、組織名、事件名、對象名。
        2. 提取名詞後，將名詞轉換延伸出相關的簡潔名詞，作為額外資訊使用。
        3. 確保每個額外資訊都只出現一次，不可以重複。
        4. 資訊要和查詢有直接關係，但不可以和查詢內容有任何重複。

        **目標查詢: {query}**
        **輸出格式:整理成列表，每行一個，不要包含編號、前言或結尾。**"""
    else:
        prompt = f"""You are an expert search optimizer.
        Please follow these steps to provide {size} additional information for each key aspect of the target query to make it easier to find relevant documents.
        
        **Steps:**
        1. Extract nouns from the target query, including but not limited to: years, months, locations, organization names, event names, object names.
        2. After extracting nouns, transform and extend them into related concise nouns to use as additional information.
        3. Ensure each piece of additional information appears only once, no repetition.
        4. The information must be directly related to the query, but cannot overlap with the query content.

        **Target Query: {query}**
        **Output Format: Organize as a list, one per line, without numbering, preamble, or conclusion.**
        """
    try:
        client = Client()
        response = client.generate(model="granite4:3b", prompt=prompt, stream=False)
        expanded_keywords = [line.strip().lstrip('0123456789.)-• ')
                    for line in response.get("response", "").split('\n')
                    if line.strip()]
        # Combine original query with expanded keywords into one string
        combined_query = query + " " + " ".join(expanded_keywords)
        return combined_query
    except Exception as e:
        print(f"Error: {e}")
        return query


def expand_query_2(query, language="en"):
    if language == "zh":
        prompt = f"""你是一個有幫助的問答助手。請回答以下問題並保留思考過程：
        {query}

        **格式:**
        reasoning:[你的思考過程]
        answer:[你的回答]"""
    else:
        prompt = f"""You are a helpful Q&A assistant. Answer the following question and keep the thinking process:
        {query}

        **Format:**
        reasoning: [Your thinking process]
        answer: [Your answer]"""
    try:
        client = Client()
        response = client.generate(model="granite4:3b", prompt=prompt, stream=False)
        # Extract only the reasoning part
        full_response = response.get("response", "")
        reasoning = ""
        if language == "zh":
            if "reasoning:" in full_response.lower():
                parts = full_response.lower().split("answer:", 1)
                reasoning = parts[0].replace("reasoning:", "").strip()
            if not reasoning:  # Fallback: use whole response
                reasoning = full_response
        else:
            if "reasoning:" in full_response.lower():
                parts = full_response.lower().split("answer:", 1)
                reasoning = parts[0].replace("reasoning:", "").strip()
            else:
                reasoning = full_response
        
        # Combine query with reasoning only
        return query + " " + reasoning
    except Exception as e:
        print(f"Error: {e}")
        return query

def expand_query_3(query, language="en"):
    pass  # Placeholder for future implementation


def retrieve_chunks(query, language="en", doc_ids=[]):
    row_chunks = get_chunks_from_db(None, doc_ids, language)
    retriever = create_retriever(row_chunks, language)
    retrieved_chunks = retriever.retrieve(query, top_k=10)
    return retrieved_chunks

def generate_answer_llm(query, retrieved_chunks, language="en"):
    from generator import generate_answer as gen_answer
    answer = gen_answer(query, retrieved_chunks, language, type="llm_chain")
    return answer