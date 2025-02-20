from transformers import AutoModelForCausalLM, AutoTokenizer

from ppo_utils.ppo_trainer_medo1 import get_reward_o1

from trl.trainer.utils import (
    truncate_response,
)

# model_name = "./ckpts/sft_stage1/checkpoint-2-4755/tfmr"
# model_name = "./ckpts/ppo_medical_o1_8B/checkpoint-157"
model_name = "./ckpts/ppo_medical_o1_8B_gpt4o/checkpoint-157"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "咳を止める方法を教えてください。"
# input_text = "88歳の女性が関節炎を抱えており、軽度の上腹部不快感を感じており、複数回コーヒーの出がらしのような物を嘔吐しています。ナプロキセンを使用していることを考慮すると、彼女の消化管の出血の最も可能性の高い原因は何ですか？"
messages = [{"role": "user", "content": input_text}]

inputs = tokenizer(
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
    return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)

# eos_token_id = tokenizer.eos_token_id
# print(f"eos_token_id: {eos_token_id}")
# if eos_token_id in outputs[0]:
#     print("eos_token_id in outputs[0]")
# else:
#     print("eos_token_id not in outputs[0]")

# print(outputs[0].tolist())
# asf

if tokenizer.eos_token_id is not None:
    # postprocessed_response = truncate_response(
    #     args.stop_token_id, processing_class.pad_token_id, response
    # )
    postprocessed_response = truncate_response(
        tokenizer.eos_token_id, tokenizer.pad_token_id, outputs
    )
    postprocessed_text = tokenizer.decode(postprocessed_response[0], skip_special_tokens=True)

    print(f"postprocessed_text: {postprocessed_text}")

output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

reward = get_reward_o1(
    model=model,
    response_ids=outputs,
    tokenizer=tokenizer,
    reward_tokenizer=None,
    pad_token_id=None,
    sub_answer=["風邪"],
)
print(f"input: {input_text}")
print(f"output: {output_text}")
print(f"reward: {reward}")
