from datasets import Dataset, Features, Sequence, Value
from langchain.embeddings.bedrock import Embeddings
from langchain.llms.amazon_api_gateway import AmazonAPIGateway
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision
)
import requests
import sys
from typing import List
from utils.langchain import LangchainLLM
from utils.answer_relevance import answer_relevancy

metrics = [
    answer_relevancy,
    faithfulness,
    context_precision
]

class AmazonAPIGatewayEmbeddings(Embeddings):
    def __init__(self, api_url, headers):
        self.api_url = api_url
        self.headers = headers

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            response = requests.post(
                self.api_url,
                json={"inputs": text},
                headers=self.headers
            )
            results.append(response.json()[0]["embedding"])

        return results

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
                self.api_url,
                json={"inputs": text},
                headers=self.headers
            )

        return response.json()[0]["embedding"]

##
# arg[1] = Data path
# arg[2] = API Url
# arg[3] = llm
# arg[4] = embeddings
# arg[5] = additional header
def get_args():
    args = []
    for arg in sys.argv:
        args.append(arg)

    if len(args) == 5:
        args.append({})
    else:
        headers = {}
        for index in range(5, len(args)):
            d = {
                args[index].split('=')[0]: args[index].split('=')[1]
            }
            headers.update(d)

        args = args[:5]
        args.append(headers)

    print(f"Arguments: {args}")

    return args

def get_llm(api_url, model_id, model_kwargs=dict(), headers=dict()):
    print(f"Get LLM {model_id}")

    llm = AmazonAPIGateway(
        api_url=f"{api_url}/invoke_model?model_id={model_id}",
        headers=headers,
        model_kwargs=model_kwargs
    )

    return llm

def get_embeddings(api_url, model_id, headers=dict()):
    print(f"Get Embeddings {model_id}")

    embeddings = AmazonAPIGatewayEmbeddings(
        api_url=f"{api_url}/invoke_model?model_id={model_id}",
        headers=headers
    )

    return embeddings

def load_data(path):
    df = pd.read_csv(
        path,
        sep=","
    )

    # renaming column ground_truth_context to contexts
    df = df.rename(columns={'ground_truth_context': 'contexts'})
    # cast type to list of string
    df['contexts'] = df['contexts'].apply(lambda x: eval(x))

    # Renaming column ground_truth to answer
    df = df.rename(columns={'ground_truth': 'answer'})

    features = {
        'question': Value(dtype='string', id=None),
        'contexts': Value(dtype='string', id=None),
        'answer': Value(dtype='string', id=None),
        'question_type': Value(dtype='string', id=None),
        'episode_done': Value(dtype='bool', id=None)
    }

    features['contexts'] = Sequence(Value(dtype='string'), length=-1, id=None)

    features = Features(features)

    dataset = Dataset.from_pandas(df, features=features)

    return dataset

if __name__ == "__main__":
    args = get_args()

    dataset = load_data(args[1])

    headers = args[-1]

    llm = get_llm(
        api_url=args[2],
        model_id=args[3],
        model_kwargs={
            "max_tokens_to_sample": 4096,
            "temperature": 0.2
        },
        headers=headers
    )

    headers["type"] = "embeddings"

    embeddings = get_embeddings(
        api_url=args[2],
        model_id=args[4],
        headers=headers
    )

    ragas_llm = LangchainLLM(llm)
    answer_relevancy.embeddings = embeddings

    for m in metrics:
        m.__setattr__("llm", ragas_llm)

    result = evaluate(
        dataset,
        metrics=metrics
    )

    print(result)
