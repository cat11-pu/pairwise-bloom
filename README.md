# bloom

一小套概率过滤器内核：定长位图、多哈希布隆过滤器、计数布隆过滤器、容量到顶后
重建的过滤器，以及按槽轮转的去重窗口。只使用 Python 标准库，键由内置的 64 位
散列函数处理，不依赖随机种子，也不做任何 I/O。

## 目录

- bloom/core.py 过滤器与位图实现
- tests/test_core.py 行为测试

## 运行测试

在项目根目录（本文件所在目录）执行：

    python3 -m unittest discover -s tests -v

全部用例通过时进程退出码为 0。若系统里没有 `python3`，把命令里的 `python3`
换成 `python` 即可（Windows 上常见）。
