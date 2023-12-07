'''
训练和测试数据格式 dataset/train.txt 和 dataset/test.txt
奇数行为输入，其下一行（偶数行）为输出
'''
import jieba

dr = jieba.get_dict_file()
d = str(dr.read(), encoding='utf-8')
words = d.splitlines()
for i, word in (enumerate(words)):
    print(word.split(' ')[0])

# jieba.load_userdict('dict.txt')
# r = '/'.join(jieba.cut('不在词典内', HMM=False))
# print(r)