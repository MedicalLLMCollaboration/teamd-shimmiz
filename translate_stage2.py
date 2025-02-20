from openai import OpenAI
import json
import os
from pydantic import BaseModel

from dotenv import load_dotenv

from tqdm import tqdm

load_dotenv()

# OpenAI APIキーを設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI()


class Data(BaseModel):
    open_ended_question: str
    ground_truth_answer: str


def translate_text(input_data):
    """
    GPTを使用してテキストを翻訳する関数。
    """

    prompt = (
        "open_ended_question,ground_truth_answerのvalueを日本語に翻訳してください。keyは英語のままで返してください。"
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
    input_data_path = "data/stage2/medical_o1_verifiable_problem.json"
    output_file = "data/stage2/medical_o1_verifiable_problem_japanese.json"

    with open(input_data_path, "r", encoding="utf-8") as f:
        input_json = json.load(f)

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            output_data_list = json.load(f)
    else:
        output_data_list = []

    # chunkごとに分ける
    bar = tqdm(total=len(input_json))
    start_i = len(output_data_list)
    for i in range(len(input_json)):
        bar.update(1)
        if i < start_i:
            continue

        input_data = input_json[i]
        output_data_raw = translate_text(input_data)

        # keyを正しいものに修正
        output_data = {
            "Open-ended Verifiable Question": output_data_raw["open_ended_question"],
            "Ground-True Answer": output_data_raw["ground_truth_answer"],
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
