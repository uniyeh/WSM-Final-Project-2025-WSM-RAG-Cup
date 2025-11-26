from ollama import Client

def generate_compositional_queries(original_query, num_queries=3, language="en"):
    if language == "zh":
        prompt = f"""你是一個專業的邏輯分析助手。你的任務是將一個**複雜的問題**拆解成 {num_queries} 個**邏輯子問題**。
        
        這與「多重查詢」不同：
        - **不要**只是換句話說（這是多重查詢的工作）。
        - **要**分析問題的邏輯結構，找出需要回答哪些「部分」才能回答整體。
        
        請遵循以下原則：
        1. **邏輯拆解**：如果問題包含多個實體或比較，請拆開來問（例如：「A和B的營收比較」 -> 「A的營收是多少？」「B的營收是多少？」）。
        2. **獨立性**：每個子問題必須是完整的句子，包含明確的實體名稱。
        3. **關鍵字保留**：確保子問題包含原始問題中的關鍵實體，以便檢索。
        
        原始問題：{original_query}
        
        請直接列出 {num_queries} 個子問題，每行一個，不要包含編號、前言或結尾。"""
    else:
        prompt = f"""You are an expert logic analyst. Your task is to break down a **complex question** into {num_queries} **logical sub-questions**.

        This is DIFFERENT from "Multi-Query":
        - **Do NOT** just rephrase the question (that is for Multi-Query).
        - **DO** analyze the logical structure and identify what "parts" need to be answered to solve the whole.

        Follow these principles:
        1. **Logical Breakdown**: If the question involves multiple entities or comparison, split them up (e.g., "Compare revenue of A and B" -> "What is the revenue of A?", "What is the revenue of B?").
        2. **Independence**: Each sub-question must be a complete sentence with specific entity names.
        3. **Keyword Preservation**: Ensure sub-questions contain the key entities from the original question for retrieval.

        Original question: {original_query}

        Provide exactly {num_queries} sub-questions, one per line, without numbering, preamble, or conclusion."""

    try:
        client = Client()
        sub_queries = client.generate(model="granite4:3b", prompt=prompt, stream=False)
        queries = [line.strip().lstrip('0123456789.)-• ')
                    for line in sub_queries.get("response", "").split('\n')
                    if line.strip()]
        return [original_query] + queries[:num_queries]
    except Exception as e:
        print(f"Error: {e}")
        return [original_query]