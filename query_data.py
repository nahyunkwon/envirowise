import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"


# Scenario #1: Home Adjustment for a Mom. Mark’s wife strug- gles to care for a 6-month-old baby and do housework. Mark wanted to upgrade his home. He scanned the rooms using AccessLens to get recommendations for common objects such as doorknobs (Fig- ure 14 1a), water faucets, and lower drawers

# ---

# implications for people with a disability or in a situational impairment. For example, turning on a manual faucet without touching it becomes useful when both of your hands are dirty, opening a pantry is helpful when you are hold- ing bags of groceries, and it is convenient to have a window that

# ---

# wall switches that can pose different contextual challenges, highlights an imperative need for solutions. Leveraging low-cost 3D-printed aug- mentations such as knob magnifiers and tactile labels seems promis- ing, yet the process of discovering unrecognized barriers remains challenging because



PROMPT_TEMPLATE = """ 
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

Follow rules below when answering.
1. If available, specify the exact resource for the solution (e.g., url for website). Do not include link if the link is not explicitly given in the context. 
2. Be specific about the reason why the suggested solution is helpful in given user characteristics in question.
3. Given several characteristics about the user, specify solution for each characteristic. 
4. Given several objects in the user's space, specify solution for each object. 
"""


def main():
    # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text

    query_text = "I have a 2-year old child. I am also using wheelchair. My kitchen has toggle light switch, microwave, fridge, drawers, etc. How can I improve my kitchen's accessibility? I want some low-cost solutions, not costly smart devices at the market. Give me urls for solutions if available."
    # query_text = "I would like to automate stuffs in my office. My office has a sink, table, stationery, monitor, keyboard, etc. How can I achieve low-cost automation without buying costly smart devices at market?"
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
