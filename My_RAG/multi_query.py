from ollama import Client

def generate_multi_queries(original_query, num_queries=3, language="en"):
    if language == "zh":
        prompt = f"""你是一個專業的搜尋優化助手。你的任務是為以下問題生成 {num_queries} 個**同義改寫**的搜尋查詢。
        
        這與「問題分解」不同：
        - **不要**拆解問題或改變問題的邏輯（這是問題分解的工作）。
        - **要**保留完整的語意，但使用不同的關鍵字來表達。
        
        請遵循以下原則以優化關鍵字匹配 (BM25)：
        1. **同義詞擴展**：使用不同的詞彙表達相同的概念（例如：「營收」vs「收入」、「獲利」vs「利潤」）。
        2. **實體保留**：絕對不要改變專有名詞和公司名稱。
        3. **句型變化**：改變問法，但問的是同一件事。
        
        原始問題：{original_query}
        
        請直接列出 {num_queries} 個查詢，每行一個，不要包含編號、前言或結尾。"""
    else:
        prompt = f"""You are an expert search optimizer. Your task is to generate {num_queries} **paraphrased variations** of the following question.

        This is DIFFERENT from "Decomposition":
        - **Do NOT** break down the question or change the logic (that is for Decomposition).
        - **DO** keep the full meaning but use different keywords.

        Follow these principles for BM25 optimization:
        1. **Synonym Expansion**: Use different vocabulary to express the same concepts (e.g., "revenue" vs "income", "profit" vs "earnings").
        2. **Entity Preservation**: NEVER change proper nouns or entity names.
        3. **Structural Variation**: Change the sentence structure, but ask for the same thing.

        Original question: {original_query}

        Provide exactly {num_queries} alternative queries, one per line, without numbering, preamble, or conclusion."""

    try:
        client = Client()
        response = client.generate(model="granite4:3b", prompt=prompt, stream=False)
        queries = [line.strip().lstrip('0123456789.)-• ')
                    for line in response.get("response", "").split('\n')
                    if line.strip()]
        return [original_query] + queries[:num_queries]
    except Exception as e:
        print(f"Error: {e}")
        return [original_query]