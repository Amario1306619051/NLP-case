import streamlit as st
from db_process.database import query_database
from LLM.model import humanize

def get_answer(question: str):
    # Query ke database
    result = query_database(
        db_path='database',
        collection_name='TESTLLMMLarge1',
        query=question
    )
    # Ambil daftar jawaban dan jarak terdekat
    ans_list = [item['answer'] for item in result['metadatas'][0]]
    min_distance = result['distances'][0]
    # Proses menjadi jawaban akhir
    final_answer = humanize(
        question=question,
        answer=ans_list,
        distance=min_distance
    )
    return final_answer

def main():
    st.set_page_config(page_title="Nawatech QA Assistant", layout="centered")
    st.title("🖥️ Nawatech QA Assistant")

    # Input pertanyaan
    q = st.text_input("Masukkan pertanyaan Anda:", "")
    
    if st.button("Cari Jawaban"):
        if not q.strip():
            st.warning("Silakan ketik pertanyaan terlebih dahulu.")
        else:
            with st.spinner("Mencari jawaban..."):
                try:
                    final = get_answer(q)
                    st.subheader("Jawaban:")
                    st.write(final)
                except Exception as e:
                    st.error(f"Gagal memproses pertanyaan: {e}")

if __name__ == "__main__":
    main()
