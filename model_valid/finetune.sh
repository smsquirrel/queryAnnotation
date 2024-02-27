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
