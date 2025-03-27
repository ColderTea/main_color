# main_color
Get theme colors from an image

可通过一张图片，提取主题色卡，获得其配色。

类 MainColor 的两个方法：
self.color_pure(self, alpha=0.75, rho_star=50)
self.color_light(self, alpha=0.33, rho_star=100)
这两个方法用于调整颜色的纯度和亮度。设置不同的 alpha 和 rho_star 可以得到不同的色调。

alpha可近似理解为调整强度：
alpha越大，调整强度越小；alpha越小，调整强度越大。

rho_star可近似理解为调整方向：
rho_star越大，调整后的纯度/亮度越高；rho_star越小，调整后的纯度/亮度越低。



