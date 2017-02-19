---
title: Python安装第三方库及常见问题
date: 2016-11-30 22:46:12
updated	: 2016-11-30 22:46:12
permalink: abc
tags:
- Python
- pip
- package
categories:
- 日志
- 运维
---

Python具有丰富的第三方库，而且绝大部分都是开源的。利用好第三方库，能避免重复造轮子，加速开发。


#### 源码安装
Python第三方库几乎都可以在github或者 pypi上找到源码。源码包格式大概有zip 、 tar.zip、 tar.bz2。解压这些包，进入解压好的文件夹，通常会有一个setup.py的文件。打开命令行，进入该文件夹。运行以下命令，就能把这个第三库安装到系统里：
```python
python setup.py install
```

或者借助pip，则不需要解压:`pip install package.zip`

#### 包管理器安装
现在很多编程语言，都带有包管理器，例如 Ruby 的 gem，nodejs的npm。

在Python中，安装第三方模块，是通过setuptools这个工具完成的。Python有两个封装了setuptools的包管理工具：easy_install和pip。目前官方推荐使用pip。

用easy_install和pip来安装第三方库很方便
它们的原理其实就是从Python的官方源pypi.python.org/pypi 下载到本地，然后解包安装。

基本操作命令如下：
```python
# 安装package
pip install packagename

# 卸载package
pip uninstall packagename

# 查看所安装的package
pip list

# 将项目依赖的库重定向输出到文件，cd到项目根目录
pip projectname > requirements.txt

# 他人安装项目的依赖库
pip install -r requirements.txt

```

pip常用命令可通过在命令行输入`pip -h`查看  
`pip command -h`可查看该命令的使用方法

```sh
Commands:
  install                     Install packages.
  download                    Download packages.
  uninstall                   Uninstall packages.
  freeze                      Output installed packages in requirements format.
  list                        List installed packages.
  show                        Show information about installed packages.
  search                      Search PyPI for packages.
  wheel                       Build wheels from your requirements.
  hash                        Compute hashes of package archives.
  completion                  A helper command used for command completion
  help                        Show help for commands.
```



#### 常见问题
1. 官方的pypi不稳定，很慢甚至访问不了

解决方法1：   
采用源码安装方式，在github或其他库下载，用python setup.py install方式安装，详见上文【源码安装】

解决方法2：  
手动指定源,在pip后面跟-i,命令如下：
```sh
pip install packagename -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
```

pipy国内镜像目前有：  
豆瓣 http://pypi.douban.com/simple/     
阿里云 http://mirrors.aliyun.com/pypi/simple/   
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/  
清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/  
华中理工大学 http://pypi.hustunique.com/    
山东理工大学  http://pypi.sdutlinux.org/   


2. 某些包在这个电脑能安装，在另一电脑有安装不了了  

参看setuptools、pip版本是否一致，升级到最新版本
```sh
pip install setuptools -U
pip install pip -U
```

3. 安装某些包时出现错误"error: Microsoft Visual C++ 10.0 is required (Unable to find vcvarsall.bat)."  

原因大概是 windows上缺少一些C编译器。

解决方法1：
安装VC或VS，该方法有时奏效，有时不奏效。

解决方法2：
更简单的解决方法：下载whl格式的package，再用pip安装。
以numpy包为例：
```sh
whl格式的下载地址：http://www.lfd.uci.edu/~gohlke/pythonlibs/
# 输入whl文件所在的完整路径
pip install D:\python\numpy-1.9.2+mkl-cp33-none-win_amd64.whl
```
