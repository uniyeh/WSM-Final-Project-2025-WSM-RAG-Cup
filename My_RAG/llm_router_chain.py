from retriever import create_retriever, get_chunks_from_db
from retriever import DenseRetriever
from generator import generate_answer
from ollama import Client

def llm_router_chain(query, language):
    query_text = query['query']['content']
    
    # 1. Do the query expansion
    # new_query = expand_query(query_text, language)
    new_query = expand_query_2(query_text, language)  # Best performing
    # new_query = expand_query_3(query_text, language)  # Dense retrieval - worse results
    print("new_query: ", new_query)
    # 2. Retrieve chunks
    retrieved_chunks = retrieve_chunks(new_query, language)  # BM25 - better results
    # retrieved_chunks = retrieve_chunks_with_dense(new_query, language)  # Dense - worse results
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
    # 1. Retrieve with Dense Retrieval
    row_chunks = get_chunks_from_db(None, [], language)  # Get all chunks
    dense_retriever = DenseRetriever(row_chunks, language=language)
    retrieved_chunks = dense_retriever.retrieve(query, top_k=5)
    scores = dense_retriever.get_scores()  # Get cached scores from retrieve()
    
    # 2. Use prompt template from paper to ask LLM to expand query
    if language == "zh":
        prompt = f"""我的目標是重新表述查詢以檢索得分高的答案文檔。

            **範例 1:**
            原始查詢：公司在2019年3月做了什麼變更？
            前3個檢索文檔：
            1. 2019年3月，公司修訂了公司治理政策...
            2. 該公司在2019年進行了多項改革...
            3. 公司治理架構在年初進行了調整...
            分數：['0.650', '0.520', '0.480']

            重新表述的查詢：[2019年3月公司治理政策修訂的具體內容]
            新分數：['0.850', '0.820', '0.780']  ← 分數提高！

            **範例 2:**
            原始查詢：誰在2020年被任命為董事？
            前3個檢索文檔：
            1. 2020年1月，James Peterson被任命為新董事...
            2. 董事會在2020年進行了人事變動...
            3. 公司在年初任命了新的董事成員...
            分數：['0.720', '0.610', '0.590']

            重新表述的查詢：[2020年1月James Peterson董事任命]
            新分數：['0.920', '0.880', '0.850']  ← 分數提高！

            ---

            **現在輪到你了:**
            查詢：{query}

            前5個檢索文檔：
            {chr(10).join([f"{i+1}. {chunk['page_content'][:200]}..." for i, chunk in enumerate(retrieved_chunks)])}

            分數：{[f'{s:.3f}' for s in scores]}

            請寫一個新的重新表述的查詢，與舊查詢不同，並且盡可能獲得高分。
            請將文本寫在方括號中。"""
    else:
        prompt = f"""My goal is to make rephrased query to retrieve answer documents with high scores.

            **Example 1:**
            Original Query: What changes did the company make in March 2019?
            TOP-3 retrieved docs:
            1. In March 2019, the company revised its corporate governance policy...
            2. The company made several reforms in 2019...
            3. Corporate governance structure was adjusted at the beginning of the year...
            Scores: ['0.650', '0.520', '0.480']

            Rephrased Query: [March 2019 corporate governance policy revision details]
            New Scores: ['0.850', '0.820', '0.780']  ← Score improved!

            **Example 2:**
            Original Query: Who was appointed as director in 2020?
            TOP-3 retrieved docs:
            1. In January 2020, James Peterson was appointed as new director...
            2. The board underwent personnel changes in 2020...
            3. The company appointed new board members at the start of the year...
            Scores: ['0.720', '0.610', '0.590']

            Rephrased Query: [January 2020 James Peterson director appointment]
            New Scores: ['0.920', '0.880', '0.850']  ← Score improved!

            ---

            **Now it's your turn:**
            Query: {query}

            TOP-5 retrieved docs:
            {chr(10).join([f"{i+1}. {chunk['page_content'][:200]}..." for i, chunk in enumerate(retrieved_chunks)])}

            Scores: {[f'{s:.3f}' for s in scores]}

            Write your new rephrased query that is different from the old ones and has a score as high as possible.
            Write the text in square brackets."""
    
    print("prompt: ", prompt)
    # 3. Return expanded query
    try:
        client = Client()
        response = client.generate(model="granite4:3b", prompt=prompt, stream=False)
        full_response = response.get("response", "").strip()
        
        # Extract text from square brackets
        import re
        match = re.search(r'\[(.*?)\]', full_response)
        if match:
            expanded_query = match.group(1).strip()
        else:
            # Fallback: use the whole response if no brackets found
            expanded_query = full_response
        return expanded_query
    except Exception as e:
        print(f"Error: {e}")
        return query

def retrieve_chunks(query, language="en", doc_ids=[]):
    row_chunks = get_chunks_from_db(None, doc_ids, language)
    retriever = create_retriever(row_chunks, language)
    retrieved_chunks = retriever.retrieve(query, top_k=10)
    return retrieved_chunks

def retrieve_chunks_with_dense(query, language="en", doc_ids=[]):
    row_chunks = get_chunks_from_db(None, doc_ids, language)
    retriever = DenseRetriever(row_chunks, language)
    retrieved_chunks = retriever.retrieve(query, top_k=10, threshold=0.3)  # Lowered threshold
    return retrieved_chunks

def generate_answer_llm(query, retrieved_chunks, language="en"):
    from generator import generate_answer as gen_answer
    answer = gen_answer(query, retrieved_chunks, language, type="llm_chain")
    return answer