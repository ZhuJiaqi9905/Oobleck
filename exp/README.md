    # 实验安排和记录

## 流程
- 把代码传输到4个节点中。检测数据集和模型参数是否存在。
    - 写脚本。
    - 模型的数据需要在aws里面下载。
- 每个节点创建8个container，修改`/etc/hosts`, `/etc/ssh/sshd_config`。启动ssh server。
    - 注意把代理的代码给删了。
    - 注意保存容器。
    - 改模型参数的时候要改cofig.json文件。注意每个节点都得改。
    - 好像tokenizer里面设置的最大长度是1024。如果用更长的需要改回去。
    
- 1个节点先ssh到其他节点上，查看ssh是否成功。
    - 写脚本。
- 进入container中，install oobleck模块。

- 跑实验，观察实验结果。获取iteration的时间。
    - 注意NCCL和GLOO的环境变量配置。
    - 写脚本。
- 根据profile的结果，获取丢失节点时传输的配置。
    - 如果离线，要修改`torch.cuda.get_device_properties("cuda:0").total_memory`
- 根据丢失节点的传输配置，获取reconfigure时间。
    - 如果离线，要修改`torch.cuda.get_device_properties("cuda:0").total_memory`

# 脚本
- `start_container.sh`。在host运行，启动容器。
- `start_container_ssh.sh`。在host运行，启动容器的ssh servers。
- `install_oobleck.sh`。在容器中运行，给其他的容器安装oobleck。
    - `local_install_oobleck.sh`。被`install_oobleck.sh`调用。
- `run_exps.py`。在容器中运行，跑实验。
- `si`
- `simulate_broadcast.py`。在容器中运行，根据丢失的layer信息计算传输时间。
