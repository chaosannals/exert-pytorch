'''
训练和测试数据格式 dataset/train.txt 和 dataset/test.txt
奇数行为输入，其下一行（偶数行）为输出
'''
import jieba

SYMBOL_END = 0
SYMBOL_START = 1 

def load_dict():
    dr = jieba.get_dict_file()
    d = str(dr.read(), encoding='utf-8')
    lines = d.splitlines()
    words = {}
    for i, line in (enumerate(lines)):
        word = line.split(' ')[0]
        words[str(i)] = word
    return words

# jieba.load_userdict('dict.txt')
# r = '/'.join(jieba.cut('不在词典内', HMM=False))
# print(r)