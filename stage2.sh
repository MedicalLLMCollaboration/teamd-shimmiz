accelerate launch \
	--num_processes 8 \
	--num_machines 1 \
	--machine_rank 0 \
    --config_file ./configs/deepspeed_zero3_stage2.yaml \
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
    --save_step 50 \
    --save_total_limit 1 \
    --eval_strategy steps \
    --eval_steps 20 \
    --kl_coef 0.05 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.05 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ppo_medical_o1_8B_gpt4o \
    --num_sample_generations 1 \
    --report_to wandb