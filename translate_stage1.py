from openai import OpenAI
import json
import os
from pydantic import BaseModel

from dotenv import load_dotenv

from tqdm import trange

load_dotenv()

# OpenAI APIキーを設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI()


class Data(BaseModel):
    question: str
    cot: str
    response: str


def translate_text(input_data):
    """
    GPTを使用してテキストを翻訳する関数。
    """

    prompt = (
        "question, cot, responseのvalueを日本語に翻訳してください。keyは英語のままで返してください。"
        f"\n{input_data}"
    )

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format=Data,
    )
    data = eval(response.choices[0].message.content)

    return data


def main():
    input_data_path = "data/stage1/medical_o1_sft.json"
    output_file = "data/stage1/medical_o1_sft_japanese.json"

    with open(input_data_path, "r", encoding="utf-8") as f:
        input_json = json.load(f)

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            output_data_list = json.load(f)
    else:
        output_data_list = []

    # chunkごとに分ける
    start_i = len(output_data_list)
    for i in trange(start_i, len(input_json)):
        input_data = input_json[i]
        output_data_raw = translate_text(input_data)

        # keyを正しいものに修正
        output_data = {
            "Question": output_data_raw["question"],
            "Complex_CoT": output_data_raw["cot"],
            "Response": output_data_raw["response"],
        }

        output_data_list.append(output_data)

        if i % 20 == 0:
            # 翻訳されたデータを保存または出力
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data_list, f, ensure_ascii=False, indent=4)

    # 翻訳されたデータを保存または出力
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
