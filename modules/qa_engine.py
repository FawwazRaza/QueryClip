import os
from dotenv import load_dotenv
from retriever import ChromaRetriever
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm_response(context, question):
    client = Groq(api_key=GROQ_API_KEY)
    model_name = "deepseek-r1-distill-llama-70b"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question using ONLY the provided context. "
                "If the answer is not present in the context, reply with 'This is Not found in the dataset.'"
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
        },
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    return chat_completion.choices[0].message.content.strip()

def main():
    TOP_K = 3

    retriever_chain = ChromaRetriever().as_retriever(search_kwargs={"k": TOP_K})

    print("Welcome to the RAG Chatbot. Type your question (or 'exit' to quit):")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        top_chunks = retriever_chain(query, k=TOP_K)
        if not top_chunks or all(not chunk["text"].strip() for chunk in top_chunks):
            print("Not found in the dataset.")
            continue

        context = "\n\n".join(
            f"[{i+1}] {chunk['text']} (Start: {chunk['start_time']}s, End: {chunk['end_time']}s)"
            for i, chunk in enumerate(top_chunks)
        )
        # context = ""
        # for i, chunk in enumerate(top_chunks):
        #     context += f"[{i+1}] {chunk['text']} (Start: {chunk['start_time']}s, End: {chunk['end_time']}s)\n\n"
        # context = context.strip()

        answer = get_llm_response(context, query)
        print("\nAnswer:", answer)
        print("\nTop retrieved chunks for transparency:")
        for i, chunk in enumerate(top_chunks):
            print(f"\nChunk {i+1}:")
            print(f"Text: {chunk['text']}")
            print(f"Start: {chunk['start_time']}s, End: {chunk['end_time']}s, Similarity: {chunk['similarity']:.3f}")

if __name__ == "__main__":
    main()
