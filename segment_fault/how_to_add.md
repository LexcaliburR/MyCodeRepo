<!--
 * @Author: lexcalibur
 * @Date: 2022-11-17 14:00:57
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2022-11-18 20:02:24
-->
1.宿主机需要运行如下命令，或者加入到.bashrc 或者 /root/.bash_profile 中
```
开启报告输出功能
sudo service apport start

设置生成core文件,core文件大小无限制
ulimit -S -c unlimited

设置core保存目录，宿主机路径
这个路径似乎只能在宿主机设置，container里面只能直接拿到这个值，所以需要挂载的时候将container里面的对应目录挂载到robo-dev的log路径下
sysctl -w kernel.core_pattern=/home/log/core.%e.%p.%t.%s

检查core文件输出路径
cat /proc/sys/kernel/core_pattern
```
2.docker启动时启用core需要添加如下命令
```
docker run -it --ulimit core=-1

```

3.使用gdb 调试错误
```
gdb [exec file] [core file]

进入gdb后，where或者bt

```