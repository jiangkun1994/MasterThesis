### Linux
- df -h：查寻磁盘空间
- du -sh：查询当前文件目录所占的大小
- sudo find / -name filename：从根目录下开始寻找这个文件
- ls | wc -l：查询当前目录下的文件数目
- diff [file1-name] [file2-name]：输出两个文件不同的地方
- tee：在终端输出信息的同时把信息记录到文件中，ls | tee ls.txt   ＃将会在终端上显示ls命令的执行结果，并把执行结果输出到ls.txt 文件中  ls | tee -a ls.txt #保留ls.txt文件中原来的内容，并把ls命令的执行结果添加到ls.txt文件的后面
### vim
- 输入“/”后加要找的关键词，输入“n”向下找关键词，“N”向上找
- 输入ggVG进行全选，gg 让光标移到首行，在vim才有效，vi中无效 
V   是进入Visual(可视）模式 
G  光标移到最后一行 
选中内容以后就可以其他的操作了，比如： 
d  删除选中内容 
y  复制选中内容到0号寄存器 
"+y  复制选中内容到＋寄存器，也就是系统的剪贴板，供其他程序用 
- chown username filename：修改文件filename的拥有权限为用户username
- export: Linux export命令用于设置或显示环境变量。export -p 列出所有的shell赋予程序的环境变量。export SQUAD_DIR=/home/jiangkun/MasterThesis/pytorch-pretrained-BERT/squad/后则可用cd $SQUAD_DIR 直接打开该对应的目录。
- echo：echo "text" >> file.txt， 将text输出到file.txt里。不加输出路径的话，默认为输出到terminal上。