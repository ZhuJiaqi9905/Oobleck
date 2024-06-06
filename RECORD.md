profile 设置2倍的内存

运行tests的方法：

`pytest --disable-warnings --log-cli-level=INFO tests/execution/test_reconfiguration.py`

一个worker对应一个gpu。

不知道agent的意义，目前一个node有一个agent。

为了让一个node上启动4个container，把profiler里面的写文件的模数改成了4。

改模型参数的时候要改cofig.json文件。注意每个节点都得改。

好像tokenizer里面设置的最大长度是1024。如果用更长的需要改回去。