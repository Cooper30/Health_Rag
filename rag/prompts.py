# rag/prompts.py
def build_prompt(query: str, docs) -> str:
    """
    app.py ile aynı mantık: context + talimat.
    LangChain zinciri kullanmadığımız için sade string döndürüyoruz.
    """
    context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    system = ("Sen bir RAG asistanısın. Sadece verilen CONTEXT'e dayanarak yanıt ver. "
              "Emin değilsen 'Bilmiyorum' de. Yanıtta kullandığın kaynak numaralarını "
              "köşeli parantezle belirt (örn. [1], [2]).")
    user = f"Soru: {query}\n\nCONTEXT:\n{context}\n\nYanıt:"
    return f"{system}\n\n{user}"
