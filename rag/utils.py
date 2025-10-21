# rag/utils.py
def keyword_filter(query: str, docs):
    """Sorgu kelimelerinden en az biri geçmiyorsa parçayı ele (hafif filtre)."""
    terms = {t for t in query.lower().replace("?", " ").replace(",", " ").split() if len(t) > 2}
    if not terms:
        return docs
    kept = [d for d in docs if any(t in d.page_content.lower() for t in terms)]
    return kept or docs  # hepsi elenirse orijinali döndür
