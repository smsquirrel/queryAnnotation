import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import logging
import torch
import numpy as np
import multiprocessing
from tools import set_seed, calculate_mrr
from models import Model
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from datasets import TextDataset, prepare_tokenizer
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer, AutoModel, AutoTokenizer, AutoConfig)

logger = logging.getLogger(__name__)


cpu_cont = 4
temperture = 1

def train(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, logger, args.train_data_file, pool, data_type="code")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=3000,
                                                num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)    
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    logger.info("  Learning rate = %f", args.learning_rate)
    
    model.zero_grad()
    if args.use_amp:
        scaler = GradScaler()
    model.train()
    tr_num,tr_loss,best_mrr=0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            code_inputs = batch[0].to(args.device)  
            if "graphcodebert" in args.model_name_or_path.lower():
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
            nl_inputs = batch[-1].to(args.device)

            with autocast(enabled=args.use_amp):
                #get code and code_block vectors
                if "graphcodebert" in args.model_name_or_path.lower():
                    func_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
                else:
                    func_vec = model(code_inputs=code_inputs)  

                nl_vec = model(nl_inputs=nl_inputs)
                loss_fct = CrossEntropyLoss()
                scores=torch.einsum("ah,bh->ab",nl_vec,func_vec)
                loss = loss_fct(scores/temperture, torch.arange(code_inputs.size(0), device=scores.device))               

            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #backward
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format(f'{args.dataset_name}_finetune_model.bin' if args.do_finetune else "model.bin")) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

            # Test best model (not need at every epoch)
            if idx == args.num_train_epochs - 1:
                result=evaluate(args, model, tokenizer,args.test_data_file, pool, eval_when_training=True)
                logger.info("***** Test results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(round(result[key],4)))


def evaluate(args, model, tokenizer,file_name,pool, eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, logger, file_name, pool, data_type="query")
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)
    
    code_dataset = TextDataset(tokenizer, args, logger, args.codebase_file, pool, data_type="code")
    print("code dataset: ", len(code_dataset))
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)    

    nl_urls, code_urls = [], []
    for example in code_dataset.examples:
        code_urls.append(example.url)
    for example in query_dataset.examples:
        nl_urls.append(example.url)
    query2CodeUrls = []
    for ids in nl_urls:
        if type(ids) == list:
            query2CodeUrls.append([code_urls.index(aid) for aid in ids])
        else:
            query2CodeUrls.append([code_urls.index(ids)])    

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs=[]
    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)
        if "graphcodebert" in args.model_name_or_path.lower():
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)


        with torch.no_grad():            
            if "graphcodebert" in args.model_name_or_path.lower():
                func_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
            else:
                func_vec = model(code_inputs=code_inputs)
            code_vecs.append(func_vec.cpu().numpy())  
 
    nl_vecs = []
    for batch in query_dataloader:  
        nl_inputs = batch[-1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy())

    model.train()
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)
    scores=np.matmul(nl_vecs,code_vecs.T)

    if query2CodeUrls is not None:
        mrr = calculate_mrr(scores, query2CodeUrls)
    
    result = {
        "eval_mrr":float(mrr)
    }

    return result

                        
                        
def main():
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_finetune", action='store_true',
                        help="Whether to run fine tuning.")                        
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--use_amp", action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    pool = multiprocessing.Pool(cpu_cont)
    
    #print arguments
    args = parser.parse_args()
    model_types = ['graphcodebert', 'codebert', 'unixcoder', 'starencoder']
    for model_type in model_types:
        if model_type in args.model_name_or_path.lower():
            args.model_type = model_type
            break
    args.dataset_name = "codesearchnet"
    dataset_names = ['codesearchnet', 'query4code', 'sods', 'conala', 'cosqa', 'staqc', "webquerytest"]
    for dataset_name in dataset_names:
        if dataset_name in args.train_data_file.lower():
            args.dataset_name = dataset_name
            break
    
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s, gpu_id: %s",device, args.n_gpu, os.environ["CUDA_VISIBLE_DEVICES"])
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_auth_token=True)
    if "starencoder" in args.model_name_or_path.lower():
        tokenizer = prepare_tokenizer(tokenizer)

    model = AutoModel.from_pretrained(args.model_name_or_path, use_auth_token=True)
    model = Model(model, args)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    
    # Training
    if args.do_train:
        if args.do_finetune:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir),strict=False)      
            model.to(args.device)        
        train(args, model, tokenizer, pool)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-mrr/{}'.format(f'{args.dataset_name}_finetune_model.bin' if args.do_finetune else "model.bin")
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.eval_data_file, pool)
        logger.info("***** Vaild results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/{}'.format(f'{args.dataset_nameS}_finetune_model.bin' if args.do_finetune else "model.bin")
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.test_data_file, pool)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

    return results


if __name__ == "__main__":
    main()

