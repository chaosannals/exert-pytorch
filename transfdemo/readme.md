# 手写 transformer 示例

此示例代码来源网络，它用 transformer 做翻译。

这个模型 只用了 一级多头注意力，而实际 transformer 是 N级。

## 多头注意力

矩阵类型：

Q 查询（矩阵）
K 键（矩阵）
KT 键（逆矩阵）
V 值（矩阵）
