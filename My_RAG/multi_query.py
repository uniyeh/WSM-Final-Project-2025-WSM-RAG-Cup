from ollama import Client

def generate_multi_queries(original_query, num_queries=3, language="en"):
     if language == "zh":
        prompt = f"""你是一個有幫助的助手，能夠生成多個搜尋查詢。
        請為以下問題生成 {num_queries} 個不同的搜尋查詢版本，這些查詢應該從不同角度表達相同的資訊需求。
        原始問題：{original_query}
        請只回傳 {num_queries} 個查詢，每行一個，不要包含編號或其他格式。"""
     else:
        prompt = f"""You are a helpful assistant that generates multiple search queries.
        Generate {num_queries} different versions of the following question to retrieve relevant documents from a vector database.
        These queries should represent the same information need but from different perspectives.
        Original question: {original_query}
        Provide only the {num_queries} alternative queries, one per line, without numbering or other formatting."""

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