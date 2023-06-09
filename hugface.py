from transformers import AutoTokenizer, AutoModel
model_directory = 'D:\huajun\chatGLM\chatglm-6B'
tokenizer = AutoTokenizer.from_pretrained(model_directory, trust_remote_code=True)
model = AutoModel.from_pretrained(model_directory, trust_remote_code=True).half().cuda()
model = model.eval()
#tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
#model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
response, history = model.chat(tokenizer, "你知道望梅止渴是什么典故么？", history=history)
print(response)
response, history = model.chat(tokenizer, "付之阙如是什么意思？", history=history)
print(response)
