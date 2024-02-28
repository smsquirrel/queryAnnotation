import json
import requests
import copy
import asyncio
import time
from abc import ABCMeta, abstractmethod
import sys 
sys.path.append('.')

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  

MinimumTrialInterval, MaximumTrialInterval, NumberOfAttempts = 10, 60, 6
RPM = 1/10
rate_limiter = asyncio.Semaphore(40)



class QAMODEL(metaclass=ABCMeta):
    def __init__(self,model_base_type="",model_name=""):
        self.model_base_type = model_base_type
        self.model_name = model_name

    def __repr__(self):
        return f"{self.model_name}"

    @abstractmethod
    def annotation(self, data:dict):
        pass


class OPENAI_MODEL(QAMODEL):
    def __init__(self, config, model_name="gpt-3.5-turbo-0613"):
        super().__init__(model_base_type="openai", model_name=model_name)
        self.config = config
        self.messages = []
        self.default_prompt = "You are a helpful assistant."
        self.OPENAI_API_KEY = [{"BASE_URL":"https://api.openai.com/v1/chat/completions", "KEY":""}]
        self.OPENAI_REQUEST_HEADER = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.OPENAI_API_KEY[0]["KEY"],
        }
        self.OPENAI_BASE_URL = self.OPENAI_API_KEY[0]["BASE_URL"]

    def get_response(self, messages):             
        # contruct prompt of openai
        time.sleep(1/RPM)
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1024,
            "stream": False,
            # "response_format":{"type":"json_object"},
            "n": self.config.candidate_nums,
        }
        try:
            try:
                response = requests.post(self.OPENAI_BASE_URL, json=request_body, headers=self.OPENAI_REQUEST_HEADER, timeout=60)
                response = json.loads(response.text)
                print("response: ", response)
                response = [response["choices"][i]["message"]["content"].strip() for i in range(self.config.candidate_nums)]
            except:
                print(f"request failed, messages: {messages}, try again...")
                if self.model_name == "gpt-4":
                    OPENAI_REQUEST_HEADER = self.OPENAI_REQUEST_HEADER
                    OPENAI_BASE_URL = self.OPENAI_BASE_URL                
                else:
                    OPENAI_REQUEST_HEADER = self.OPENAI_REQUEST_HEADER
                    OPENAI_BASE_URL = self.OPENAI_BASE_URL
                try:
                    response = requests.post(OPENAI_BASE_URL, json=request_body, headers=OPENAI_REQUEST_HEADER, timeout=90)
                    response = json.loads(response.text)
                    print("response: ", response)
                    response = [response["choices"][i]["message"]["content"].strip() for i in range(self.config.candidate_nums)]
                except:
                    if self.model_name == "gpt-4":
                        OPENAI_REQUEST_HEADER = copy.deepcopy(self.OPENAI_REQUEST_HEADER)
                        OPENAI_REQUEST_HEADER["Authorization"] = "Bearer " + self.OPENAI_API_KEY[-1]["KEY"]
                        OPENAI_BASE_URL = self.OPENAI_API_KEY[-1]["BASE_URL"]
                    else:
                        OPENAI_REQUEST_HEADER = self.OPENAI_REQUEST_HEADER
                        OPENAI_BASE_URL = self.OPENAI_BASE_URL     
                    response = requests.post(OPENAI_BASE_URL, json=request_body, headers=OPENAI_REQUEST_HEADER, timeout=90)
                    response = json.loads(response.text)
                    response = [response["choices"][i]["message"]["content"].strip() for i in range(self.config.candidate_nums)]           
            if self.config.candidate_nums == 1:
                response = response[0]
        except:
            response = None       
        return response
    
    @retry(wait=wait_random_exponential(min=MinimumTrialInterval, max=MaximumTrialInterval), stop=stop_after_attempt(NumberOfAttempts))
    async def request(self, data):
        async with rate_limiter:
            loop = asyncio.get_event_loop()
            messages = [{"role": "system", "content": self.default_prompt}]
            messages += [{"role": "user", "content": data}]
            response = await loop.run_in_executor(None, self.get_response, messages)
            return response

    @retry(wait=wait_random_exponential(min=MinimumTrialInterval, max=MaximumTrialInterval), stop=stop_after_attempt(NumberOfAttempts))
    async def checkScore(self, data, examples=None):
        async with rate_limiter:
            loop = asyncio.get_event_loop()
            messages = [{"role": "system", "content": self.config.prompts["check_prompt"]}]
            if examples is not None:
                for example in examples:
                    messages += [{"role": "user", "content": "Code:\n" + example["code"]}, {"role": "assistant", "content": "Query:" + example["query"]}]
            messages += [{"role": "user", "content": "API: " + data["api"] + "\nDocument Explanation: " + data["document_func"] + "\nUser Description: " + data["model_output"]}]   
            query = await loop.run_in_executor(None, self.get_response, messages)
            return query        
        
    @retry(wait=wait_random_exponential(min=MinimumTrialInterval, max=MaximumTrialInterval), stop=stop_after_attempt(NumberOfAttempts))
    async def annotation(self, data, summary=None, examples=None):
        async with rate_limiter:
            loop = asyncio.get_event_loop()
            if summary is not None:
                system_prompt = self.config.prompts["query_generation_prompt"]
                prompt = "Code:\n" + data["func"] + "\nCode Explanation: " + summary
            else:
                system_prompt = self.config.prompts["direct_query_prompt"]
                prompt = "Code:\n" + data["function"] 
            messages = [{"role": "system", "content": system_prompt}]

            # if examples is not None:
            #     for example in examples:
            #         messages += [{"role": "user", "content": "Code:\n" + example["code"]}, {"role": "assistant", "content": "Query:" + example["query"]}]
            messages += [{"role": "user", "content": prompt}]  
            query = await loop.run_in_executor(None, self.get_response, messages)
            return query
        
    @retry(wait=wait_random_exponential(min=MinimumTrialInterval, max=MaximumTrialInterval), stop=stop_after_attempt(NumberOfAttempts))
    async def summary_generation(self, data, intra_repo_calls=None, api_calls=None, examples=None):
        async with rate_limiter:
            loop = asyncio.get_event_loop()
            # print(f"data:{data}")
            messages = [{"role": "system", "content": self.config.prompts["summary_prompt"]}]
            # if examples is not None:
            #     for example in examples:
            #         messages += [{"role": "user", "content": "Code:\n" + example["code"]}, {"role": "assistant", "content": "Query:" + example["query"]}]
            summary_generate_prompt = "Code:\n" + data["function"]
            if intra_repo_calls is not None:
                summary_generate_prompt += "\nIntra Repository Calls:\n" + "\n".join(intra_repo_calls)
            if api_calls is not None:
                summary_generate_prompt += "\nThird party API calls:\n" + "\n".join(api_calls)
            messages += [{"role": "user", "content": summary_generate_prompt}]   
            summary = await loop.run_in_executor(None, self.get_response, messages)
            return summary
        
    @retry(wait=wait_random_exponential(min=MinimumTrialInterval, max=MaximumTrialInterval), stop=stop_after_attempt(NumberOfAttempts))
    async def verify(self, data, examples=None):
        async with rate_limiter:
            loop = asyncio.get_event_loop()
            # print(f"data:{data}")
            messages = [{"role": "system", "content": self.config.prompts["verification_prompt"]}]
            if examples is not None:
                for example in examples:
                    messages += [{"role": "user", "content": "Code:\n" + example["code"]}, {"role": "assistant", "content": "Query:" + example["query"]}]
            messages += [{"role": "user", "content": data["query"][0] + "\nCode:\n" + data["function"]}]   
            summary = await loop.run_in_executor(None, self.get_response, messages)
            return summary

class ChatGPT(OPENAI_MODEL):
    def __init__(self, config):
        super().__init__(config, model_name="gpt-3.5-turbo-0613")


class GPT4(OPENAI_MODEL):
    def __init__(self, config):
        super().__init__(config, model_name="gpt-4-1106-preview")
