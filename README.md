## Enhancing Code Retrieval with High-Quality Dataset Generation through Large Language Models


### Framework
![framework picture](./framework.png "Framework")

### Parse repository
To parse the repository, you first need to clone the repository content. Then you can analyze the function call situation by running the following command:
```bash
cd parse_repo
python parse_repo.py
```

### Run Annotation
To run data annotation, you need to fill in the openai key and execute the following command:
```bash
cd annotation
python annotation.py
```

### Run Pretraining
You can run `bash pretrain.sh` to finetune model. We use the CodeSearchNet dataset to pre-train the CodeBERT model as an example:
```bash
lang=python
model=codebert
train_dataset=Q4C

declare -A model_mapping
model_mapping["codebert"]="microsoft/codebert-base"
model_mapping["graphcodebert"]="microsoft/graphcodebert-base"
model_mapping["unixcoder"]="microsoft/unixcoder-base"
model_mapping["starencoder"]="bigcode/starencoder"

model_name_or_path=${model_mapping[$model]}
mkdir -p ./saved_models_${model}_${train_dataset}
python main.py \
    --output_dir=./saved_models_${model}_${train_dataset} \
    --config_name=${model_name_or_path} \
    --model_name_or_path=${model_name_or_path} \
    --tokenizer_name=${model_name_or_path} \
    --lang=$lang \
    --do_train \
    --use_amp \
    --train_data_file=dataset/${train_dataset}/train.jsonl \
    --eval_data_file=dataset/${train_dataset}/valid.jsonl \
    --test_data_file=dataset/${train_dataset}/test.jsonl \
    --codebase_file=dataset/${train_dataset}/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 1e-5 \
    --seed 123456 2>&1| tee saved_${model}_${train_dataset}/pretrain.log
```

### Run Fine-tuning
You can run `bash finetune.sh` to finetune model. We use the StaQC dataset to finetune the pre-trained CodeBERT model as an example:
```bash
lang=python
model=codebert
train_dataset=Q4C

declare -A model_mapping
model_mapping["codebert"]="microsoft/codebert-base"
model_mapping["graphcodebert"]="microsoft/graphcodebert-base"
model_mapping["unixcoder"]="microsoft/unixcoder-base"
model_mapping["starencoder"]="bigcode/starencoder"

model_name_or_path=${model_mapping[$model]}
mkdir -p ./saved_models_${model}_${train_dataset}
python main.py \
    --output_dir=./saved_models_${model}_${train_dataset} \
    --config_name=${model_name_or_path} \
    --model_name_or_path=${model_name_or_path} \
    --tokenizer_name=${model_name_or_path} \
    --lang=$lang \
    --do_train \
    --do_finetune \
    --use_amp \
    --train_data_file=dataset/staqc/train_pair.jsonl \
    --eval_data_file=dataset/staqc/test_pair.jsonl \
    --test_data_file=dataset/staqc/test_pair.jsonl \
    --codebase_file=dataset/staqc/train_pair.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 1e-5 \
    --seed 123456 2>&1| tee saved_${model}_${train_dataset}/finetune_staqc.log
```


### Evaluation
You need to run `bash eval.sh` to evaluate model performance on all benchmarks. We use the CosQA dataset to finetune the pre-trained CodeBERT model as an example:
```bash
lang=python
model=codebert
train_dataset=Q4C

declare -A model_mapping
model_mapping["codebert"]="microsoft/codebert-base"
model_mapping["graphcodebert"]="microsoft/graphcodebert-base"
model_mapping["unixcoder"]="microsoft/unixcoder-base"
model_mapping["starencoder"]="bigcode/starencoder"

model_name_or_path=${model_mapping[$model]}

python main.py \
    --output_dir=./saved_models_${model}_${train_dataset}/$lang \
    --config_name=${model_name_or_path} \
    --model_name_or_path=${model_name_or_path} \
    --tokenizer_name=${model_name_or_path} \
    --lang=$lang \
    --do_test \
    --use_amp \
    --train_data_file=dataset/cosqa/train_pair.jsonl \
    --eval_data_file=dataset/cosqa/valid_pair.jsonl \
    --test_data_file=dataset/cosqa/test_pair.jsonl \
    --codebase_file=dataset/cosqa/code_base.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 1e-5 \
    --seed 123456 2>&1| tee saved_${model}_${train_dataset}/$lang/eval_cosqa.log
```

### Dataset
The annotated Query4Code dataset will be made public in the end.
