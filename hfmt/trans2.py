from transformers import MarianMTModel, MarianTokenizer

path = 'hfmt/model/zh2en'
tokenizer = MarianTokenizer.from_pretrained(path)
model = MarianMTModel.from_pretrained(path)
tokens = tokenizer.prepare_seq2seq_batch([
    '测试',
    '中文翻译',
    '机器学习',
    '深度学习',
])
translated = model.generate(**tokens)
result = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(result)
