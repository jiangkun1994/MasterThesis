### Notes
- 安装pytorch0.3.0和cuda8：conda install pytorch=0.3.0 torchvision cuda80 -c pytorch
- 将安装好的cuda8与导入的pytorch成功运行：export LD_LIBRARY_PATH=/home/jiangkun/anaconda3/pkgs/cudatoolkit-8.0-1/lib（可以先用find / -name libcudart.so.8.0找到路径）
- 在windows利用anaconda安装tensorflow时，只能建立在python3.x的版本上，因此python2版本在windows上无法安装tensorflow
- [配置谷歌云计算平台GCE](https://zhuanlan.zhihu.com/p/33099231)，[本地ubuntu连接GCE](https://www.jianshu.com/p/57e85cf3e50b)
- [Stanford CoreNLP在ubuntu下的安装与使用](https://blog.csdn.net/Hay54/article/details/82313535)，安装好java环境之后，将相关jar文件添加入CLASSPATH即可
- **np.sum(a, axis=0, keepdims=True)**：axis=0是指X轴，即矩阵a的每一列的和，而axis=1则指Y轴，即矩阵a每一行的和。keepdims=True表示做完和以后，结果依然保持原来的矩阵格式，不会变成np.array这样形式的数组
- **np.random.rand vs np.random.randn**：前者数值服从均匀分布，后者数值服从正态分布
- **显卡**：（GPU）主流是Nvidia的GPU，深度学习本身需要大量计算。GPU的并行计算能力，在过去几年里恰当地满足了深度学习的需求。AMD的GPU基本没有什么支持，可以不用考虑。
- **驱动**：没有显卡驱动，就不能识别GPU硬件，不能调用其计算资源。
- **CUDA**：是Nvidia推出的只能用于自家GPU的并行计算框架。只有安装这个框架才能够进行复杂的并行计算。主流的深度学习框架也都是基于CUDA进行GPU并行加速的，几乎无一例外。还有一个叫做cudnn，是针对深度卷积神经网络的加速库。
- `urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name r`：解决办法，在程序开始加上：`from urllib.request import urlopen`
- FileZilla上传文件至GCE：[Google Cloud FTP Setup with FileZilla ](https://www.onepagezen.com/google-cloud-ftp-filezilla-quick-start/)。Windows下该key文件在E盘的putty文件夹里的key1，用户名为rsa-key-20190114。


### HotpotQA
- **测试集**：对于该数据集的Test set in the fullwiki setting，该测试集只包含了question和basic IR model检索出来的对应每个问题的paragraphs，并没有答案和相对应的supporting facts，因此只需要用改进后的IR model检索出来的paragraphs来代替之前的paragraphs后，对测试集里的问题的答案和supporting facts进行预测，将结果上传到作者提供的evaluation server即可。
- **JSON Format**：[{_id:, question:, answer:, supporting_facts:, context:}, {_id:, question:, answer:, supporting_facts:, context:}, {_id:, question:, answer:, supporting_facts:, context:} .......]
- **Supporting Facts**：[[title, sent_id], [title, sent_id], [title, sent_id] ........]
- **Context**：[[title, sentences], [title, sentences], [title, sentences] ......]，其中，sentences = [string1, string2, string3, ......]

- **利用DrQA里的TF-IDF model对来自HotpotQA和SQuAD进行检索测试**:
数据格式脚本见[这里](./scripts/DrQA_eval_txt.py)
```
python scripts/retriever/eval.py ~/DrQA/data/datasets/hotpotreduced-dev.txt # HotpotQA
python scripts/retriever/eval.py ~/DrQA/data/datasets/stanford-dev.txt # SQuAD
```
### DrQA
- **SQuAD-v1.1-train.json**：对于SQuAD的json文件，它只有一行，是一个dict，包含了很多数据点
```
with open('./SQuAD-v1.1-train.json') as f:
    dataset = json.load(f)

for line in open('./SQuAD-v1.1-train.json'):
    dataset = json.loads(line)
```
