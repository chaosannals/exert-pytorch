from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

path = 'hfmt/model/zh2en'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSeq2SeqLM.from_pretrained(path)
tokens = tokenizer.prepare_seq2seq_batch([
    '测试',
    '中文翻译',
    '机器学习',
    '深度学习',
])
translated = model.generate(**tokens)
result = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(result)
