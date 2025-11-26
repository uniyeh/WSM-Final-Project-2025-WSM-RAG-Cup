python3 My_RAG/main.py --query_path ./dragonball_dataset/test_queries_zh.jsonl --docs_path ./dragonball_dataset/dragonball_docs.jsonl --language zh --output ./predictions/predictions_zh.jsonl
python3 My_RAG/main.py --query_path ./dragonball_dataset/test_queries_en.jsonl --docs_path ./dragonball_dataset/dragonball_docs.jsonl --language en --output ./predictions/predictions_en.jsonl


python3 rageval/evaluation/main.py --input_file ./predictions/predictions_zh.jsonl --output_file ./result/score_zh.jsonl --language zh
python3 rageval/evaluation/main.py --input_file ./predictions/predictions_en.jsonl --output_file ./result/score_en.jsonl --language en

python3 rageval/evaluation/process_intermediate.py

# Multi query
python3 My_RAG/main.py --query_path ./dragonball_dataset/test_queries_en.jsonl --docs_path ./dragonball_dataset/dragonball_docs.jsonl --language en --output ./predictions/predictions_en_multi.jsonl --use_multi_query
python3 rageval/evaluation/main.py --input_file ./predictions/predictions_en_multi.jsonl --output_file ./result/score_en_multi.jsonl --language en

# Decomposition query
python3 My_RAG/main.py --query_path ./dragonball_dataset/test_queries_en.jsonl --docs_path ./dragonball_dataset/dragonball_docs.jsonl --language en --output ./predictions/predictions_en_decomposition.jsonl --use_compositional_query
python3 rageval/evaluation/main.py --input_file ./predictions/predictions_en_decomposition.jsonl --output_file ./result/score_en_decomposition.jsonl --language en
