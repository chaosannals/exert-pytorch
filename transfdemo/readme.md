# 手写 transformer 示例

示例代码来源网络，它用 transformer 做翻译。
只需要把训练资料里，翻译结果换做下一句回答，就成了聊天模型。

示例手动编码了 9 个词。需要改用一个已有大辞典的编码。

这个模型 只用了 一级多头注意力，而实际 transformer 是 N级。

1. 靠预测词素集里出现结束词来结束输出，代码给出。
2. 靠预测词素集出现开始词来增长输出，只有开始词定义，没有实现继续输出。TODO

## 多头注意力

矩阵类型：

Q 查询（矩阵）
K 键（矩阵）
KT 键（逆矩阵）
V 值（矩阵）
