# install python packages Python包安装指南

### open cmd in windows 

cd C:\Users\User\AppData\Local\Programs\Python\Python35-32\

python -m pip install numpy

python -m pip install pandas

python -m pip install "D:\Download\Python\lxml-3.6.1-cp35-cp35m-win32.whl" lxml-3.6.1-cp35-cp35m-win32.whl

python -m pip install tushare

改目录先改系统盘目录

e:

cd C:\Users\AppData\Local\Programs\Python\Python35-32\

# 
## anaconda 在线安装包

打开Anaconda prompt

输入

pip install sklearn-pandas

or

conda install xgboost.tar.bz2

# 
## anaconda 离线安装包

安装包从anaconda下载win64的tar.bz2文件

打开Anaconda prompt

输入

conda install C:\路径\sklearn-pandas.tar.bz2

打开python，检查是否已成功安装，不报错：

import xgboost

检查安装的包的版本：

xgboost.__version__

## anaconda 离线安装包 方法二

在线环境下打开Anaconda prompt

输入

pip install sklearn-pandas

or

conda install xgboost

安装好后，在D:\...\Anaconda\Lib\site-packages\中找到对应的两个文件夹

并在D:\...\Anaconda\conda-meta\中找到对应的json文件

复制出来，上传到离线环境中

把对应的文件放到对应的site-packages和conda-meta文件夹中

打开python，检查是否已成功安装，不报错：

import xgboost

检查安装的包的版本：

xgboost.__version__

注意：要把在线环境下用Anaconda prompt自动安装的所有包都复制过去。有些包没有json文件。

如果import后报错缺某个包，则把对应的包再复制上去



