# 实验安排和记录

## 流程
- 注意profile相关的代码
- 注意构建pipeline template相关的代码
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
    - 在`tmp/logs/install`文件夹下会显示每个容器的安装进度。
- `run_exps.py`。在容器中运行，跑实验。
- `kill.sh`杀死进程。在master节点上kill进程后，child节点的进程也会自动退出。
- `simulate_broadcast.py`。在容器中运行，根据丢失的layer信息计算传输时间。

# 简易流程：
- 假设基础环境已经搭建好了。
- 所有脚本都是在工作目录的根目录下运行。并且在运行的shell里面需要先进行`conda activate oobleck`切换到conda环境。
- 如果修改了代码，就git push上去。然后每个节点git pull。再在master容器里运行`./exp/install_oobleck.sh`。从`tmp/logs/install`文件夹下看安装进度。
- 如果安装完毕。就运行`python ./exp/run_exps.py`跑实验。在`tmp/logs`里面有实验结果。
- 如果中途想退出，就Ctrl-C。然后在master节点执行下`./exp/kill.sh`。
- 还有一种跑实验的方法，可以在调试的时候用。就是先运行`./master.sh`脚本，它会阻塞的启动master。大概等5~10秒后。在再另一个shell窗口运行`./job.sh`来启动任务。这个适合在调试某一个配置的时候用。这样kill master的时候，所有的进程也就都kill了。