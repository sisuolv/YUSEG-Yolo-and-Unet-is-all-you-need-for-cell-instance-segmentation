# Base Images
## 从天池基础镜像构建(from的base img 根据自己的需要更换，建议使用天池open list镜像链接：https://tianchi.aliyun.com/forum/postDetail?postId=67720)
FROM  registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.7-cuda11.0-py3

COPY  requirements_delete.txt requirements.txt
##安装依赖包,pip包请在requirements.txt添加
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=3600 && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=3600

## 把当前文件夹里的文件构建到镜像的根目录下
WORKDIR /workspace
COPY ./   /workspace
## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）

## 镜像启动后统一执行 sh run.sh
#CMD ["sh", "run.sh"]
