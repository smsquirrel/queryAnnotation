from tqdm import tqdm
import pickle

def countPop():
    API2Pop = {}

    with open("./all_repo2functions.pkl", "rb") as f:
        all_repo2functions = pickle.load(f)

    for repo_name, functions in tqdm(all_repo2functions.items(), desc="count API popularity..."):
        for function in functions:
            for call in function['third_party_calls']:
                if call['identifier'] not in API2Pop:
                    API2Pop[call['identifier']] = 0
                API2Pop[call['identifier']] += 1

    with open("./API2Pop.pkl", "wb") as f:
        pickle.dump(API2Pop, f)


if __name__ == "__main__":
    countPop()