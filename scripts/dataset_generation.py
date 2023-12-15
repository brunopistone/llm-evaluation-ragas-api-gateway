from langchain.document_loaders import WebBaseLoader
from langchain_core.embeddings import Embeddings
from langchain.llms.amazon_api_gateway import AmazonAPIGateway
import requests
import sys
from typing import List
from ragas.llms import LangchainLLM
from ragas.testset import testset_generator

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
# arg[1] = API Url
# arg[2] = llm generator
# arg[3] = llm critic
# arg[4] = embeddings
# arg[5] = threshold
# arg[6] = test size
# arg[7] = additional header
def get_args():
    args = []
    for arg in sys.argv:
        args.append(arg)

    if len(args) == 7:
        args.append({})
    else:
        headers = {}
        for index in range(7, len(args)):
            d = {
                args[index].split('=')[0]: args[index].split('=')[1]
            }
            headers.update(d)

        args = args[:7]
        args.append(headers)

    print(f"Arguments: {args}")

    return args

def get_llm(api_url, model_id, model_kwargs=dict(), headers=dict()):
    llm = AmazonAPIGateway(
        api_url=f"{api_url}/invoke_model?model_id={model_id}",
        headers=headers,
        model_kwargs=model_kwargs
    )

    return llm

def get_embeddings(api_url, model_id, headers=dict()):
    embeddings = AmazonAPIGatewayEmbeddings(
        api_url=f"{api_url}/invoke_model?model_id={model_id}",
        headers=headers
    )

    return embeddings

##
# Change this function if you want to access data from a different source
def load_data():
    url = [
        "https://aws.amazon.com/bedrock/",
        "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html",
        "https://aws.amazon.com/blogs/aws/preview-enable-foundation-models-to-complete-tasks-with-agents-for-amazon-bedrock/",
        "https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html",
        "https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html",

    ]

    loader = WebBaseLoader(url)

    data = loader.load()

    return data

if __name__ == "__main__":

    args = get_args()

    data = load_data()

    headers = args[7]

    llm_generator = get_llm(
        api_url=args[1],
        model_id=args[2],
        model_kwargs={
            "max_tokens_to_sample": 4096,
            "temperature": 0.2
        },
        headers=headers
    )

    llm_critic = get_llm(
        api_url=args[1],
        model_id=args[3],
        model_kwargs={
            "max_tokens_to_sample": 4096,
            "temperature": 0.2
        },
        headers=headers
    )

    headers["type"] = "embeddings"

    embeddings = get_embeddings(
        api_url=args[1],
        model_id=args[4],
        headers=headers
    )

    testsetgenerator = testset_generator.TestsetGenerator(
        generator_llm=LangchainLLM(llm=llm_generator),
        critic_llm=LangchainLLM(llm=llm_critic),
        embeddings_model=embeddings
    )

    testsetgenerator.threshold = float(args[5])

    print("Start dataset generation...")
    testset = testsetgenerator.generate(data, test_size=int(args[6]))
    print("Test dataset generated!")

    df = testset.to_pandas()

    df.dropna(inplace=True)
    df = df[df.ne('').all(1)]

    df.to_csv(
        "./dataset.csv",
        index=False,
        header=True,
        encoding="utf-8",
        sep=","
    )

    df.to_json(
        "./dataset.json",
        index=False
    )
