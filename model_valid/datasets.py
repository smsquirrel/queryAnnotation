import numpy as np
import torch
import pickle
import os
import json
from tqdm import tqdm
# from datasets import load_dataset
from torch.utils.data import Dataset
from tree_sitter import Language, Parser
from transformers import AutoTokenizer
from parser.utils import extract_dataflow
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript



dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens=None,
                 code_ids=None,
                 position_idx=None,      
                 dfg_to_code=None,
                 dfg_to_dfg=None,       
                 nl_tokens=None,
                 nl_ids=None,
                 url=None,
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url

class convert_examples_to_features:
    def __init__(self, lang="python", model_type="codebert", dataset_name="codesearchnet"):
        self.lang = lang
        self.model_type = model_type
        self.dataset_name = dataset_name
    
    def __call__(self, item):
        js,tokenizer,args=item
        try:
            if self.dataset_name not in ["codesearchnet"]:
                code = js["code"]
            else:
                if self.model_type == "graphcodebert":
                    code = js['original_string']
                else:
                    code = ' '.join(js['code_tokens'])
        except:
            code = ""
        if self.model_type=="graphcodebert":
            if code != "":
                #extract data flow
                code_tokens,dfg=extract_dataflow(code,parsers[lang], lang)
                code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
                ori2cur_pos={}
                ori2cur_pos[-1]=(0,0)
                for i in range(len(code_tokens)):
                    ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
                code_tokens=[y for x in code_tokens for y in x]  
                #truncating
                code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]
                code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]

                code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
                position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]   # pad_token_id is 1, start from 2
                dfg=dfg[:args.code_length+args.data_flow_length-len(code_tokens)]
                code_tokens+=[x[0] for x in dfg]
                position_idx+=[0 for x in dfg]   # dfg token position is 0
                code_ids+=[tokenizer.unk_token_id for x in dfg]
                padding_length=args.code_length+args.data_flow_length-len(code_ids)
                position_idx+=[tokenizer.pad_token_id]*padding_length  # extra postion id is 1
                code_ids+=[tokenizer.pad_token_id]*padding_length    
                #reindex
                reverse_index={}
                for idx,x in enumerate(dfg):
                    reverse_index[x[1]]=idx
                for idx,x in enumerate(dfg):
                    dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
                dfg_to_dfg=[x[-1] for x in dfg]
                dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
                length = len([tokenizer.cls_token])
                dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
            else:
                code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg = [], [], [], [], []
        elif self.model_type in ["codebert", "starencoder"]:
            code_tokens = tokenizer.tokenize(code)
            code_tokens = code_tokens[:args.code_length-2]
            code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
            # add padding
            padding_length = args.code_length - len(code_tokens)
            code_tokens += [tokenizer.pad_token]*padding_length
            code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
            position_idx = [i + tokenizer.pad_token_id + 1 for i in range(args.code_length)]    
            dfg_to_code, dfg_to_dfg = None, None    
        elif self.model_type == "unixcoder":
            code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
            code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
            code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
            padding_length = args.code_length - len(code_ids)
            code_ids += [tokenizer.pad_token_id]*padding_length
            position_idx = [i + tokenizer.pad_token_id + 1 for i in range(args.code_length)]
            dfg_to_code, dfg_to_dfg = None, None
        else:
            raise ValueError("model type not supported, must in [codebert, starencoder, graphcodebert, unixcoder]")
            
        if self.dataset_name in ["python_adv", "codesearchnet"]:
            nl = ' '.join(js['docstring_tokens'])
        else:
            nl = js.get("query", "")
        if self.model_type=="unixcoder":
            nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
            nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
            nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
            padding_length = args.nl_length - len(nl_ids)
            nl_ids += [tokenizer.pad_token_id]*padding_length  
        else:
            nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
            nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
            nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
            padding_length = args.nl_length - len(nl_ids)
            nl_ids+=[tokenizer.pad_token_id]*padding_length      
        url = js.get("url", js.get("relevant_ids"))
        return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg,nl_tokens,nl_ids,url)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_path=None,pool=None, data_type="code") -> None:
        self.args=args
        prefix=file_path.split('/')[-1][:-6]
        cache_file=args.output_dir+'/'+self.args.dataset_name+"_"+prefix+'.pkl'
        self.data_type = data_type

        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
        else:
            self.examples = []
            data=[]
            idx = 0
            with open(file_path) as f:
                for line in f:
                    idx += 1
                    line=line.strip()
                    js=json.loads(line)
                    data.append((js,tokenizer,args))
            self.examples=pool.map(convert_examples_to_features('python', model_type=args.model_type, dataset_name=args.dataset_name), tqdm(data,total=len(data)))
            pickle.dump(self.examples,open(cache_file,'wb'))
            
        if data_type == "code":
            for idx, example in enumerate(self.examples[:5]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))          
                logger.info("url: {}".format(example.url))      


    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        if self.args.model_type == "graphcodebert" and self.data_type == "code":
            #calculate graph-guided masked function
            attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length,
                                self.args.code_length+self.args.data_flow_length),dtype=bool)
            #calculate begin index of node and max length of input
            node_index=sum([i>1 for i in self.examples[item].position_idx])   # code length
            max_length=sum([i!=1 for i in self.examples[item].position_idx])  # code + dfg length
            #sequence can attend to sequence
            attn_mask[:node_index,:node_index]=True
            #special tokens attend to all tokens
            for idx,i in enumerate(self.examples[item].code_ids):
                if i in [0,2]:
                    attn_mask[idx,:max_length]=True
            # add all language special tokens attention
            for idx,i in enumerate(self.examples[item].position_idx):
                if i in [2]:
                    attn_mask[idx,:max_length]=True
            #nodes attend to code tokens that are identified from
            for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
                if a<node_index and b<node_index:
                    attn_mask[idx+node_index,a:b]=True   # dfg to code edge
                    attn_mask[a:b,idx+node_index]=True   # code to dfg edge
            #nodes attend to adjacent nodes 
            for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
                for a in nodes:
                    if a+node_index<len(self.examples[item].position_idx):
                        attn_mask[idx+node_index,a+node_index]=True   # 为何使用的单向的边
            attn_mask = torch.tensor(attn_mask)
        else:
            attn_mask = torch.tensor([])

        if self.args.model_type == "graphcodebert":
            return (torch.tensor(self.examples[item].code_ids),
                attn_mask,
                torch.tensor(self.examples[item].position_idx), 
                torch.tensor(self.examples[item].nl_ids))
        else:
            return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(self.examples[item].position_idx), 
                torch.tensor(self.examples[item].nl_ids))


def prepare_tokenizer(tokenizer):
    # Special tokens
    MASK_TOKEN = "<mask>"
    SEPARATOR_TOKEN = "<sep>"
    PAD_TOKEN = "<pad>"
    CLS_TOKEN = "<cls>"

    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.add_special_tokens({"sep_token": SEPARATOR_TOKEN})
    tokenizer.add_special_tokens({"cls_token": CLS_TOKEN})
    tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return tokenizer