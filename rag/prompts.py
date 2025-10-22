#  rag/prompts.py
# Amaç:
# - Kullanıcıdan gelen soru (query) ve FAISS aramasından dönen doküman parçalarını
#   birleştirerek LLM'e (Gemini) gönderilecek nihai prompt metnini üretmek.
# - Bu dosya, RAG mimarisinde "Prompt Engineering" katmanını temsil eder.
# - Prompt yapısı sabittir: sistem rolü + context + kullanıcı sorusu.

def build_prompt(query: str, docs) -> str:
    """
    Soru ve context parçalarından LLM için prompt oluşturur.

    Args:
        query (str): Kullanıcı tarafından sorulan soru.
        docs (list): FAISS aramasından dönen doküman objeleri (LangChain Document).

    Returns:
        str: Gemini LLM'e gönderilecek nihai prompt metni.

    Açıklama:
    - Context: Doküman içerikleri numaralandırılarak birleştirilir.
    - System rolü: "Sadece context'e dayanarak yanıt ver" kuralı belirtilir.
    - Yanıtta kaynak numaralarının [1], [2] formatında yer alması istenir.
    - LangChain zinciri kullanılmadığı için sade bir string döndürülür.
    """
    # Context dokümanlarını numaralandırarak tek bir string haline getir
    context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

    # Sistem rolü (talimat kısmı)
    system = (
        "Sen bir RAG asistanısın. Sadece verilen CONTEXT'e dayanarak yanıt ver. "
        "Emin değilsen 'Bilmiyorum' de. Yanıtta kullandığın kaynak numaralarını "
        "köşeli parantezle belirt (örn. [1], [2])."
    )

    # Kullanıcı sorusu + context birleştirilerek prompt tamamlanır
    user = f"Soru: {query}\n\nCONTEXT:\n{context}\n\nYanıt:"
    return f"{system}\n\n{user}"
