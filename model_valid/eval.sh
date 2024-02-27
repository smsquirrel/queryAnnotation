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
