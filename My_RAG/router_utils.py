import re
import os
import sys
from ollama import Client
# Add parent directory to path to import Connection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../db')))
from Connection import Connection
from utils import load_ollama_config

DB_PATH = "db/dataset.db"

def specific_router(query):
    content = query['query']['content']
    
    # Medical: Hospitalization records
    # Pattern 1: "the hospitalization record of [Name]."
    match1 = re.search(r"the hospitalization record of (.*?)\.", content)
    if match1:
        name = match1.group(1)
        domain, doc_ids = search_db_by_content(name, "Medical")
        if not doc_ids:
            return "Medical", []
        return domain, doc_ids
        
    # Pattern 3: "Based on the hospitalization record of [Name],"
    match3 = re.search(r"Based on the hospitalization record of (.*?),", content)
    if match3:
        name = match3.group(1)
        domain, doc_ids = search_db_by_content(name, "Medical")
        if not doc_ids:
            return "Medical", []
        return domain, doc_ids
        
    # Pattern 2: "which hospitals were [Name] and [Name] admitted to"
    match2 = re.search(r"which hospitals were (.*?) admitted to", content, re.IGNORECASE)
    if match2:
        names_str = match2.group(1)
        names = [n.strip() for n in re.split(r'\s+and\s+|,\s*', names_str) if n.strip()]
        all_doc_ids = []
        for name in names:
            _, doc_ids = search_db_by_content(name, "Medical")
            all_doc_ids.extend(doc_ids)
        if all_doc_ids:
            return "Medical", list(set(all_doc_ids))
        # If we matched the pattern but found no docs, it's still Medical
        return "Medical", []
            
    # Finance: Asset acquisition
    # Pattern: "acquisition of [Company] completed" or "acquisition of [Company] in"
    match_finance = re.search(r"acquisition of (.*?) (?:completed|in)", content, re.IGNORECASE)
    if match_finance:
        company = match_finance.group(1)
        domain, doc_ids = search_db_by_content(company, "Finance")
        if not doc_ids:
            return "Finance", []
        return domain, doc_ids

    # Finance: Based on ... report
    # Pattern: "Based on [Company]'s ... report" or "Based on [Company] and [Company]'s ... report"
    match_report = re.search(r"Based on (.*?)'s? .*?report", content, re.IGNORECASE)
    if match_report:
        companies_str = match_report.group(1)
        # Split by "and", ",", "&"
        companies = [c.strip() for c in re.split(r'\s+and\s+|,\s*|&', companies_str) if c.strip()]
        all_doc_ids = []
        for company in companies:
            # Clean up "Ltd." etc if needed, but search_db_by_content handles stemming
            _, doc_ids = search_db_by_content(company, "Finance")
            all_doc_ids.extend(doc_ids)
        
        if all_doc_ids:
            return "Finance", list(set(all_doc_ids))
        return "Finance", []

    # Finance: Fallback for "Ltd." or "Inc."
    if "Ltd." in content or "Inc." in content or "Corp." in content:
         # Try to extract the company name using LLM or just return Finance if we can't find docs
         # For now, let's rely on the extract_search_terms fallback below, 
         # BUT if that returns nothing, we should default to Finance because of the explicit indicator.
         pass

    # Chinese Patterns
    
    # Medical: Hospitalization records (Chinese)
    # Pattern: "根据[Name]的住院病历" or "根据[Name]和[Name]的住院病历"
    match_zh_medical = re.search(r"根据(.*?)的住院病历", content)
    if match_zh_medical:
        names_str = match_zh_medical.group(1)
        # Split by "和" or "、" or space
        names = [n.strip() for n in re.split(r'和|、|\s+', names_str) if n.strip()]
        all_doc_ids = []
        for name in names:
            _, doc_ids = search_db_by_content(name, "Medical")
            all_doc_ids.extend(doc_ids)
        
        if all_doc_ids:
            return "Medical", list(set(all_doc_ids))
        return "Medical", []

    # Law: Judgments (Chinese)
    # Pattern: "根据[Name]的判决书" or "根据[Name]和[Name]的判决书"
    match_zh_law = re.search(r"根据(.*?)的判决书", content)
    if match_zh_law:
        names_str = match_zh_law.group(1)
        # Split by "和" or "、" or space
        names = [n.strip() for n in re.split(r'和|、|\s+', names_str) if n.strip()]
        all_doc_ids = []
        for name in names:
            _, doc_ids = search_db_by_content(name, "Law")
            all_doc_ids.extend(doc_ids)
            
        if all_doc_ids:
            return "Law", list(set(all_doc_ids))
        return "Law", []

    # Finance: Acquisition (Chinese)
    # Pattern: "[Company]的收购" (Acquisition of [Company]) or "收购[Company]"
    match_zh_finance = re.search(r"收购(.*?)(?:完成|的)?", content)
    if match_zh_finance:
        company = match_zh_finance.group(1)
        # Clean up company name if it captured too much
        if company.endswith("的"):
            company = company[:-1]
            
        domain, doc_ids = search_db_by_content(company, "Finance")
        if not doc_ids:
            return "Finance", []
        return domain, doc_ids

    # Finance: Established year (Chinese)
    # Pattern: "[Company]成立于哪一年"
    match_zh_est = re.search(r"(.*?)公司成立于哪一年", content)
    if match_zh_est:
        company = match_zh_est.group(1)
        company = company + "公司"
        domain, doc_ids = search_db_by_content(company, "Finance")
        if not doc_ids:
            return "Finance", []
        return domain, doc_ids

    # Finance: Financial Report (Chinese)
    # Pattern: "根据[Company]...的财务报告"
    match_zh_report = re.search(r"根据(.*?)(?:[0-9]{4}年)?的财务报告", content)
    if match_zh_report:
        company = match_zh_report.group(1)
        domain, doc_ids = search_db_by_content(company, "Finance")
        if not doc_ids:
            return "Finance", []
        return domain, doc_ids

    terms_str = extract_search_terms(content)
    # print(f"[terms_str] query: {query['query']['query_id']}, with terms_str: {terms_str}, with content: {query['query']['content']}")
    if terms_str:
        terms = [t.strip() for t in terms_str.split(',') if t.strip()]
        if terms:
            # Try searching with all terms
            domain, doc_ids = search_db_by_content(terms)
            if doc_ids:
                return domain, doc_ids
    return None, []
    # Fallback: Use LLM to extract search terms for specific queries
    # This handles cases like "Who was appointed as Chief Operating Officer in October 2021?"
    # We only try this if the query seems specific (contains dates or specific roles)
    if any(x in content.lower() for x in ["appointed", "acquisition", "hospitalization", "record"]):
        terms_str = extract_search_terms(content)
        print(f"[terms_str] query: {query['query']['query_id']}, with terms_str: {terms_str}, with content: {query['query']['content']}")
        if terms_str:
            terms = [t.strip() for t in terms_str.split(',') if t.strip()]
            if terms:
                # Try searching with all terms
                domain, doc_ids = search_db_by_content(terms)
                if doc_ids:
                    return domain, doc_ids

    return None, []

def extract_search_terms(query):
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    prompt = f"""Identify the most specific search terms in the following query. 
    Extract names, roles, companies, dates, key events, specific actions, and significant noun phrases.
    Return ONLY a comma-separated list of terms. 
    Do not include generic words like "who", "what", "when", "where".
    IMPORTANT: Separate month and year into different terms (e.g. "October 2021" -> "October", "2021").
    
    Query: {query}
    
    Search Terms:"""
    response = client.generate(model=ollama_config["model"], prompt=prompt, stream=False)
    content = response.get("response", "").strip()
    return content
    

def search_db_by_content(keywords, domain=None):
    conn = Connection(DB_PATH)
    
    if isinstance(keywords, str):
        keywords = [keywords]
    
    # Clean keywords
    keywords = [k.strip() for k in keywords if k.strip()]
    if not keywords:
        return None, []

    # Stem keywords for matching
    stemmed_keywords = [simple_stem(k) for k in keywords]

    # Strategy: OR search to get candidates, then rank by sentence-level overlap
    query_sql = "SELECT doc_id, domain, content FROM documents WHERE "
    conditions = []
    params = []
    
    for keyword in keywords:
        conditions.append("content LIKE ?")
        params.append(f'%{keyword}%')
    
    query_sql += " OR ".join(conditions)
    
    if domain:
        query_sql = f"SELECT doc_id, domain, content FROM ({query_sql}) WHERE domain = ?"
        params.append(domain)
        
    cursor = conn.execute(query_sql, tuple(params))
    rows = cursor.fetchall()
    
    if not rows:
        return None, []
        
    # Rank results
    scored_docs = []
    for row in rows:
        doc_id = row[0]
        doc_domain = row[1]
        content = row[2]
        
        # Split into sentences (simple split by period)
        sentences = content.split('.')
        
        max_sentence_score = 0
        
        for sentence in sentences:
            # Tokenize sentence
            words = re.findall(r'\w+', sentence.lower())
            stemmed_words = [simple_stem(w) for w in words]
            
            score = 0
            matches = 0
            
            # Check for keyword matches in this sentence
            # We check if the stemmed keyword is contained in the stemmed words of the sentence
            # Or if the keyword phrase appears in the sentence
            
            for i, keyword in enumerate(keywords):
                stemmed_keyword = stemmed_keywords[i]
                
                # Check for phrase match in original sentence (case insensitive)
                if keyword.lower() in sentence.lower():
                    matches += 1
                # Check for stemmed word match
                elif ' ' not in keyword and stemmed_keyword in stemmed_words:
                    matches += 1
            
            # Score = matches^2 to reward density
            if matches > 0:
                score = matches * matches
                if score > max_sentence_score:
                    max_sentence_score = score
        
        scored_docs.append((max_sentence_score, doc_domain, doc_id))
        
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Filter to get top scores
    if not scored_docs:
        return None, []
        
    max_score = scored_docs[0][0]
    
    # If max_score is 0 (shouldn't happen if SQL returned rows, but possible if stemming logic differs), return empty
    if max_score == 0:
        return None, []

    # Return all docs with max_score
    best_docs = [d for d in scored_docs if d[0] == max_score]
    
    found_domain = domain if domain else best_docs[0][1]
    doc_ids = [d[2] for d in best_docs]
    
    return found_domain, doc_ids


def simple_stem(word):
    """Simple stemming to handle common suffixes."""
    word = word.lower()
    if word.endswith('s') and not word.endswith('ss'):
        word = word[:-1]
    elif word.endswith('ed'):
        word = word[:-2]
    elif word.endswith('ing'):
        word = word[:-3]
    elif word.endswith('ment'):
        word = word[:-4]
    elif word.endswith('ion'):
        word = word[:-3]
    elif word.endswith('ly'):
        word = word[:-2]
    return word
