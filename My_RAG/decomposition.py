from ollama import Client

def generate_compositional_queries(original_query, num_queries=3, language="en"):
     if language == "zh":
        prompt = f"""你是一個有幫助的助手，能夠生成多個搜尋查詢。
        請為以下問題生成更簡單、更具體的子問題，這些子問題應該從不同角度表達相同的資訊需求。
        原始問題：{original_query}
        請只回傳 {num_queries} 個子問題，每行一個，不要包含編號或其他格式。"""
     else:
        prompt = f"""You are a helpful assistant. Break down the following complex question into simpler, more specific sub-questions.
        These sub-questions should help answer the original question when combined.
        Original question: {original_query}
        Provide only {num_queries} sub-questions, one per line, without numbering."""

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