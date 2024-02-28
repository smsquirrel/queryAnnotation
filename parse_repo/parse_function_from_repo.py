import os
import pickle
import json
import pandas as pd
import tempfile
from tqdm import tqdm
from tree_sitter import Language
from language_data import LANGUAGE_METADATA
from process import DataProcessor

language = 'python'
DataProcessor.PARSER.set_language(Language('./my-languages.so', language))  



processor = DataProcessor(language=language,
                          language_parser=LANGUAGE_METADATA[language]['language_parser'])


def collect_repo_name(filtered_data_dir="./codesearchnet_python/"):
    with open(os.path.join(filtered_data_dir, "train.jsonl"), "rb") as f:
        codesearchnet_filtered = [json.loads(line) for line in f]
    with open(os.path.join(filtered_data_dir, "codebase.jsonl"), "rb") as f:
        codesearchnet_filtered += [json.loads(line) for line in f]
    return codesearchnet_filtered

def from_datasetlist_to_dict(dataset_list):
    repo2functions = {}
    for data in dataset_list:
        if data['repo'] not in repo2functions:
            repo2functions[data['repo']] = []
        repo2functions[data['repo']].append(data)
    return repo2functions



def parser_repository_files(repo_name, repo_hash, temp_repo_path, repo_dirs):
    functions = processor.process_dee(repo_name, ext=LANGUAGE_METADATA[language]['ext'], tmp_dir=temp_repo_path, sha=repo_hash)
    calls, edges, third_party_calls = processor.process_dent(repo_name, ext=LANGUAGE_METADATA[language]['ext'], library_candidates=functions, tmp_dir=temp_repo_path, sha=repo_hash, repo_dirs=repo_dirs)
    return functions, calls, edges, third_party_calls


def get_all_repo2functions(repo2file, repo2hash):
    all_repo2functions = {}
    i = 0
    for repo_name, files in tqdm(list(repo2file.items()), desc="parse repo..."):
        if i > 5:
            break
        else:
            i += 1
        repo_hash = repo2hash[repo_name]
        with tempfile.TemporaryDirectory() as temp_repo_path:
            print(f"Create repo: {repo_name} temporary directory in {temp_repo_path}")
            repo_dirs = []
            for file_name, content in files.items():
                file_path = os.path.join(temp_repo_path, file_name)
                for repo_dir in file_name.split('/')[:-1]:
                    repo_dirs.append(repo_dir)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create parent directories if they don't exist
                with open(file_path, "w") as file:
                    file.write(content)
            repo_dirs = list(set(repo_dirs))

            functions, calls, edges, third_party_calls = parser_repository_files(repo_name, repo_hash, temp_repo_path, repo_dirs)     
            for function in functions:
                function['repo'] = repo_name

            for function in functions:
                function['calls'] = []
                for call in calls:
                    if function['identifier'] == call['main_function'] and function["path"] == call['path']:
                        function['calls'].append({"identifier": call['identifier'], "argument_list": call["argument_list"]})

            for function in functions:
                function['third_party_calls'] = []
                for third_party_call in third_party_calls:
                    if function['identifier'] == third_party_call['main_function'] and function["path"] == third_party_call['path']:
                        function['third_party_calls'].append({"identifier": third_party_call['identifier'], "argument_list": third_party_call["argument_list"]})
            all_repo2functions[repo_name] = functions
            functions = pd.DataFrame(functions)
            functions = functions[["identifier", "path", "docstring_tokens", "function"]]
            print(f"parse functions: {functions}")

        # break
        return all_repo2functions

def remove_empty_functions(all_repos2functions):
    remove_nums = 0
    for repo_name, functions in tqdm(all_repos2functions.items()):
        origin_function_num = len(functions)
        all_repos2functions[repo_name] = [function for function in functions if function['function'] != ""]
        remove_num = len(all_repos2functions[repo_name]) - origin_function_num
        remove_nums += remove_num
        if remove_num > 0:
            print(f"repo: {repo_name} remove {remove_num} empty functions")
    print(f"remove {remove_nums} empty functions")
    return all_repos2functions

def annotation_flag(all_repos2functions):
    """According to the data in CodeSearchNet, for functions in all_repos2functions, label them as 1 if they exist in CodeSearchNet; otherwise, label them as 0."""

    original_codesearch_dataset = collect_repo_name(f"../../codesearchnet_{language}/")
    original_repo2function = from_datasetlist_to_dict(original_codesearch_dataset)
    for repo_name, functions in tqdm(all_repos2functions.items()):
        for function in functions:
            function['annotation'] = 0
            for data in original_repo2function[repo_name]:
                if function['identifier'] == data['func_name'] and function['path'] == data['path']:
                    function['annotation'] = 1
                    break

    return all_repos2functions


if __name__ == "__main__":
    with open("./all_repos2functions_withflag.pkl", "rb") as f:
        all_repo2functions = pickle.load(f)
    annotation_flag(all_repo2functions)
    