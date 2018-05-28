## 系统安装恢复
### 双系统重装windows后恢复grub
```
1、制作grub for dos的启动U盘
2、U盘启动
3、grub> root (hd1,10) #各系统不一样
   grub> kernel /vmlinuz root=/dev/sdaX #可用ls命令查看root下的文件；cat查看/etc/fstab确定root在X
   grub> initrd /initrd.img
   grub> boot  #进入linux
   
 4、$ sudo update-grub #可以发现windows
    $ sudo grub install /dev/grab #安装grub到硬盘
 5、重启
```
