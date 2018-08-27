/*install python packages*/
/*open cmd in windows*/
cd C:\Users\User\AppData\Local\Programs\Python\Python35-32\
python -m pip install numpy
python -m pip install pandas
python -m pip install "D:\Download\Python\lxml-3.6.1-cp35-cp35m-win32.whl" lxml-3.6.1-cp35-cp35m-win32.whl
python -m pip install tushare

#改目录先改系统盘目录
e:
cd C:\Users\AppData\Local\Programs\Python\Python35-32\

#anaconda 在线安装
pip install sklearn-pandas

#anaconda 离线安装
#安装包从anaconda下win64的tar.bz2文件
conda install C:\Users\AppData\Local\Programs\Python\Python35-32\sklearn-pandas.tar.bz2
conda install xgboost.tar.bz2

#或者本地在线安装好后，从Lib目录下拷出来整个文件夹
#从安装包里的setup.py可以看到依赖包

#XGBoost包:
py-xgboost-0.72-py36h6538335_0.tar.bz2
libxgboost-0.72-0.tar.bz2
m2w64_***.tar.bz2
