import os
import json
import pickle
import asyncio
import random
from tqdm import tqdm
from LLMs import ChatGPT, GPT4

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

models = {"ChatGPT":ChatGPT, "GPT4":GPT4}

class Config():
    def __init__(self, model="ChatGPT", few_shot=0, prompts="", device="cuda", candidate_nums=1, mode="direct_annotation", verification=False):
        self.model = model
        self.prompts = prompts
        self.device = device
        self.few_shot = few_shot
        self.candidate_nums = candidate_nums
        self.call_relationship = False
        self.API_call = False
        self.mode = mode
        self.verification = verification

    def __str__(self):
        return f"model: {self.model} shot: {self.few_shot}\n prompts: {self.prompts}\n candidate_nums: {self.candidate_nums} mode: {self.mode} verification: {self.verification}"


def collect_repo_name(filtered_data_dir="../codesearchnet_python/"):
    with open(os.path.join(filtered_data_dir, "train.jsonl"), "rb") as f:
        codesearchnet_filtered = [json.loads(line) for line in f]
    with open(os.path.join(filtered_data_dir, "codebase.jsonl"), "rb") as f:
        codesearchnet_filtered += [json.loads(line) for line in f]
    return codesearchnet_filtered


def topological_sort(graph):
    def dfs(node):
        visited[node] = 1
        for neighbor in range(len(graph)):
            if graph[node][neighbor] == 1: 
                if visited[neighbor] == 1:
                    # Remove the edge to break the cycle
                    graph[node][neighbor] = 0
                elif visited[neighbor] == 0:  # If neighbor is not visited
                    if not dfs(neighbor):
                        return False
        visited[node] = 2
        result.append(node)
        return True

    num_nodes = len(graph)
    visited = [0] * num_nodes  
    result = []

    for node in range(num_nodes):
        if visited[node] == 0:
            if not dfs(node):
                return None  # Graph has a cycle

    return result[::-1]


async def annotate_repo(graph, repo_funcs, api_details):
    sorted_nodes = topological_sort(graph)
    
    model = models[config.model](config)
    repo_summaries = []
    for node_id in sorted_nodes:
        intra_repo_calls = [repo_funcs[call] for call in range(len(repo_funcs)) if graph[node_id][call] == 1]
        api_calls = [api_details[api] for api in repo_funcs[node_id]["third_party_calls"] if api in api_details]
        summary = await model.summary_generation(repo_funcs[node_id], intra_repo_calls=intra_repo_calls, api_calls=api_calls)   
        repo_summaries.append(summary)
    
    return repo_summaries


async def summary_annotation(config, datasets):
    with open("./api_details.json", "r") as f:
        api_details = json.load(f)
    tasks = []
    for repo_name, repo_funcs in datasets.items():
        Graph = [[0 for _ in range(len(repo_funcs))] for _ in range(len(repo_funcs))]
        for idx, func in enumerate(repo_funcs):
            for call in func["calls"]:
                if call in repo_funcs:
                    Graph[idx][repo_funcs.index(call)] = 1
        tasks.append(annotate_repo(graph=Graph, repo_funcs=repo_funcs, api_details=api_details))
    repo_summaries = await asyncio.gather(*tasks)        
    all_summaries = [summary for repo_summary in repo_summaries for summary in repo_summary]
    return all_summaries
                

async def annotation_score(config, data_path):
    with open(data_path, "r") as f:
        datasets = [json.loads(line) for line in f] 

    # datasets = datasets[:3]       
    prefix = data_path.split("/")[-1].split(".")[0]
    model = models[config.model](config)
    tasks = [model.verify(data) for data in datasets]

    responses = await asyncio.gather(*tasks)

    with open(f"./{prefix}_annotation_results.pkl", "wb") as f:
        pickle.dump((datasets, responses), f)


async def main(config):
    step = 500
    with open("../all_repos2functions_withflag.pkl", "rb") as f:
        repo2function = pickle.load(f)
    # dict to list
    datasets = [repo_funcs for repo_name, repo_funcs in repo2function.items()]

    datasets = [func for repo_funcs in datasets for func in repo_funcs if func['annotation'] == 1]
    print(f"Test dataset nums:{len(datasets)}, data:{datasets[0]}")
    model = models[config.model](config)
    if config.mode == "direct_annotation":
        tasks = [model.annotation(data) for data in datasets]
        querys = await asyncio.gather(*tasks)
    else:
        summaries = await summary_annotation(config, repo2function)
        for i in range(0, len(datasets), step):
            print(f"Start get query {i} to {i+step}...")
            tasks = [model.annotation(data, summary) for data, summary in zip(datasets[i:i+step], summaries[i:i+step])]
            querys = await asyncio.gather(*tasks)

            with open(f"./synthesized_dataset/query/annotation_{i}.jsonl", "w") as f:
                for summary, query in zip(summaries, querys):
                    f.write(json.dumps({'summary': summary, "query": query}) + "\n")          

        # verification_results = await annotation_score(config, "./synthesized_dataset/query/annotation_0.jsonl")      
    print("Annotating done!")


def mergeDataset(method_prefix="two_step_annotation"):
    datas = []
    for file in os.listdir("./synthesized_dataset/query/"):
        if file.startswith(method_prefix):
            with open(f"./synthesized_dataset/query/{file}", "r") as f:
                datas += [json.loads(line) for line in f]
    
    with open("./synthesized_dataset/Query4Code.jsonl", "w") as f:
        for data in datas:
            f.write(json.dumps(data) + "\n")



if __name__ == "__main__":
    direct_query_prompt = """Please act as a query generator. 
    For the given function-level code in the repository, please provide a query that the user might use. This query should be able to search for that function in a search engine. 
    Note that you should not provide any other information."""
    direct_examples = [{"code": "def func(a, b):\n\treturn a + b", "query": "python function add two numbers"}]
    summary_prompt = """Please play the role of a programming expert. 
    For the functions in the given repository by users, please provide a detailed summary of their functionalities. 
    Please note that you need to provide a concise summary of the code instead of explaining it step by step, and you do not need to reply with any other information."""    # Note that you should not reply with any other information.
    check_prompt = """I hope you can play the role of a programming expert. For a given API and its corresponding documentation explanation, as well as a user's description of the API's functionality, please help me confirm the degree to which the user-provided description of the API's functionality matches with what is described in the documentation. If it completely matches semantically, award 2 points; if it partially matches, give 1 point; if there is no match, give 0 points.
    Please provide the reasons for your rating and the final score.
    Please reply in JSON format, including two fields: "reason", "score"."""
    query_generation_prompt = """Please play the role of a programmer who is searching for code. 
    For a function-level code and its functional summary (to help you understand the function's purpose) provided by the user, please provide a query, which can be used to search for that function on a search engine.
    Please note, reply in JSON format, which includes a field "Query"."""
    verification_prompt = """Please play the role of a programming expert. For the given user queries and function pairs, please judge whether the code can meet the needs of the user's query based on the following principles:
1. The code can answer and exceed the requirements for query needs (3 points);
2. The code can satisfy a certain category of query needs (2 points);
3. The code only meets less than 50% of query needs (1 points);
4. The code is only minimally related to the query (0 point).
Please provide an explanation along with corresponding scores, noting that you need to output in JSON format as follows: `{"explanation": <explanation>, "related_score": <score>}`, without providing any other information"""
    prompts = {"direct_query_prompt": direct_query_prompt, "summary_prompt": summary_prompt, "check_prompt": check_prompt, "query_generation_prompt": query_generation_prompt, "verification_prompt": verification_prompt}
    config = Config(model="ChatGPT", prompts=prompts, few_shot=0, candidate_nums=1, mode="two_step_annotation", verification=False)    # direct_annotation, two_step_annotation
    print("Start annotating, config: ", config)
    asyncio.run(main(config))
