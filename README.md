# 医療コンペo1

## 環境
- GCP H100 * 8台
  - disk size: 1000GB
  - CUDA: 12.4
  - Driver: 550.90.07

## インストール
1. [uv](https://docs.astral.sh/uv/) のインストール
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. ライブラリのインストール
```
uv sync
```

3. 環境の有効化
```
source .venv/bin/activate
```


## SFT (Stage1)
### 英語ver
1. データの準備
[medical_o1_sft.json](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/blob/main/medical_o1_sft.json) をダウンロードして以下のpathに置く
```
data/stage1/medical_o1_sft.json
```

2. 学習
- 学習時間 1エポック13分、全部で40分くらい
```
accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 1 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard SFT_stage1.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --data_path data/stage1/medical_o1_sft.json
```

### 日本語ver

1. データの準備
英語verの1と一緒

2. 日本語に翻訳(事前に実行済み、スキップでOK)
以下のコマンドで翻訳
```
python3 translate_stage1.py
```
以下のpathに日本語のdataが出力される
```
data/stage1/medical_o1_sft_japanese.json
```

3. 学習
- wandbで連携してログを確認するためにコマンドを叩く
```
wandb login
```
- 学習step 全部で3epoch
    - 1epoch 13分くらい
```
accelerate launch --config_file ./configs/deepspeed_zero3_stage1.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard SFT_stage1.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --data_path data/stage1/medical_o1_sft_japanese.json
```

- output
```
./ckpts/sft_stage1/checkpoint-2-4755/tfmr
```

## RL(Stage2)
### 英語ver
1. データの準備
[.json](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem/blob/main/medical_o1_verifiable_problem.json) をダウンロードして以下のpathにおく
```
data/stage2/medical_o1_verifiable_problem.json
```

2. 学習
```
accelerate launch \
	--num_processes 8 \
	--num_machines 1 \
	--machine_rank 0 \
  --config_file ./configs/deepspeed_zero3_stage2.yaml \
	--deepspeed_multinode_launcher standard RL_stage2.py \
  --model_name_or_path FreedomIntelligence/HuatuoGPT-o1-8B \
  --reward_model_path FreedomIntelligence/medical_o1_verifier_3B \
  --value_model_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_name data/stage2/medical_o1_verifiable_problem.json \
  --response_length 1300 \
  --temperature 0.5 \
  --local_rollout_forward_batch_size 8 \
  --num_ppo_epochs 3 \
  --num_mini_batches 1 \
  --total_episodes 20000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 True \
  --output_dir ./ckpts \
  --save_strategy steps \
  --save_step 20 \
  --save_total_limit 1 \
  --eval_strategy steps \
  --eval_steps 20 \
  --kl_coef 0.03 \
  --learning_rate 5e-7 \
  --warmup_ratio 0.05 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --run_name ppo_medical_o1_8B \
  --num_sample_generations -1 \
  --report_to wandb
```


### 日本語ver
1. データの準備
英語版と一緒

2. 日本語に翻訳(事前に実行済み、スキップでOK)
```
python3 translate_stage2.py
```
以下のpathに日本語のdataが出力される
```
data/stage2/medical_o1_verifiable_problem_japanese.json
```

3. 学習
```
accelerate launch \
	--num_processes 8 \
	--num_machines 1 \
	--machine_rank 0 \
  --config_file ./configs/deepspeed_zero3.yaml \
	--deepspeed_multinode_launcher standard RL_stage2.py \
  --model_name_or_path ./ckpts/sft_stage1/checkpoint-2-4755/tfmr \
  --reward_model_path FreedomIntelligence/medical_o1_verifier_3B \
  --value_model_path Qwen/Qwen2.5-3B-Instruct \
  --dataset_name data/stage2/medical_o1_verifiable_problem_japanese.json \
  --response_length 1300 \
  --temperature 0.5 \
  --local_rollout_forward_batch_size 8 \
  --num_ppo_epochs 3 \
  --num_mini_batches 1 \
  --total_episodes 20000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 True \
  --output_dir ./ckpts \
  --save_strategy steps \
  --save_step 100 \
  --save_total_limit 1 \
  --eval_strategy steps \
  --eval_steps 20 \
  --kl_coef 0.03 \
  --learning_rate 5e-7 \
  --warmup_ratio 0.05 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --run_name ppo_medical_o1_8B \
  --num_sample_generations -1 \
  --report_to wandb
```
