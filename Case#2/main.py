import streamlit as st
from db_process.database import query_database
from LLM.model import humanize

def get_answer(question: str):
    # Query the database
    result = query_database(
        db_path='database',  # Path to the database
        collection_name='TESTLLMMLarge1',  # Collection name in the database
        query=question  # User's question
    )
    # Extract the list of answers and the closest distance
    ans_list = [item['answer'] for item in result['metadatas'][0]]  # List of answers
    min_distance = result['distances'][0]  # Minimum distance
    # Process the answers into a final response
    final_answer = humanize(
        question=question,  # Original question
        answer=ans_list,  # List of answers
        distance=min_distance  # Closest distance
    )
    return final_answer

def main():
    # Configure the Streamlit page
    st.set_page_config(page_title="Nawatech QA Assistant", layout="centered")
    st.title("🖥️ Nawatech QA Assistant")  # Page title

    # Input field for the user's question
    q = st.text_input("Masukkan pertanyaan Anda:", "")  # "Enter your question"

    # Button to trigger the answer search
    if st.button("Cari Jawaban"):  # "Search Answer"
        if not q.strip():  # Check if the input is empty
            st.warning("Silakan ketik pertanyaan terlebih dahulu.")  # "Please type your question first."
        else:
            with st.spinner("Mencari jawaban..."):  # "Searching for an answer..."
                try:
                    final = get_answer(q)  # Get the final answer
                    st.subheader("Jawaban:")  # "Answer:"
                    st.write(final)  # Display the answer
                except Exception as e:
                    st.error(f"Gagal memproses pertanyaan: {e}")  # "Failed to process the question"

if __name__ == "__main__":
    main()
