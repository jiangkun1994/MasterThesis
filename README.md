### Notes
- 安装pytorch0.3.0和cuda8：conda install pytorch=0.3.0 torchvision cuda80 -c pytorch
- 将安装好的cuda8与导入的pytorch成功运行：export LD_LIBRARY_PATH=/home/jiangkun/anaconda3/pkgs/cudatoolkit-8.0-1/lib（可以先用find / -name libcudart.so.8.0找到路径）
- 在windows利用anaconda安装tensorflow时，只能建立在python3.x的版本上，因此python2版本在windows上无法安装tensorflow
- [配置谷歌云计算平台GCE](https://zhuanlan.zhihu.com/p/33099231)，[本地ubuntu连接GCE](https://www.jianshu.com/p/57e85cf3e50b)
- [Stanford CoreNLP在ubuntu下的安装与使用](https://blog.csdn.net/Hay54/article/details/82313535)，安装好java环境之后，将相关jar文件添加入CLASSPATH即可

### HotpotQA
- **测试集**：对于该数据集的Test set in the fullwiki setting，该测试集只包含了question和basic IR model检索出来的对应每个问题的paragraphs，并没有答案和相对应的supporting facts，因此只需要用改进后的IR model检索出来的paragraphs来代替之前的paragraphs后，对测试集里的问题的答案和supporting facts进行预测，将结果上传到作者提供的evaluation server即可。
- **JSON Format**：[{_id:, question:, answer:, supporting_facts:, context:}, {_id:, question:, answer:, supporting_facts:, context:}, {_id:, question:, answer:, supporting_facts:, context:} .......]
- **Supporting Facts**：[[title, sent_id], [title, sent_id], [title, sent_id] ........]
- **Context**：[[title, sentences], [title, sentences], [title, sentences] ......]，其中，sentences = [string1, string2, string3, ......]