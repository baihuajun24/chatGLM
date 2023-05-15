from transformers import AutoTokenizer, AutoModel
import pandas as pd


# model_directory = 'D:/huajun/chatGLM/chatglm-6B'
model_directory = 'D:\huajun\chatGLM\chatGLM\chatglm-6B'
tokenizer = AutoTokenizer.from_pretrained(model_directory, trust_remote_code=True)
model = AutoModel.from_pretrained(model_directory, trust_remote_code=True).half().cuda()
model = model.eval()

df = pd.read_json('classification/data/train.json', lines=True)
output_prediction_path = 'classification/data/output_prediction_zeroshot.txt'
with open('classification/data/label.txt', 'r') as f:
    label_list = [line.strip() for line in f]
label_string_list = ', '.join(label_list)
corrected = 0
num_preview= 10
with open (output_prediction_path, 'w') as f:
    for index, row in df[:num_preview].iterrows():
        # query = f'完成论文多分类任务，类别有这些：{label_string_list}. 只用中文输出标签. \n 已知论文标题为{row["title"]}，摘要内容为{row["abstract"]}'
        query = f'完成论文多分类任务，类别有这些：{label_string_list}. 用中文输出标签，用冒号给出分类理由. \n 已知论文标题为{row["title"]}'
        response, history = model.chat(tokenizer, query, history=[])
        print(f'title: {row["title"]}, response: {response}, actual label: {row["subject_name"][0]}')
        f.write(f'{row["id"], response}\n')
        if row["subject_name"][0] == response:
            corrected += 1

print(f'Total correct guesses: {corrected}. Accuracy: {corrected/num_preview}')
