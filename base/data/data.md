Pytorch做数据增强的方法是在原始图片上进行的，并覆盖原始图片

ToTensor处理过程

1.对数据进行transpose，原来是h*w*c，会变成c*h*w，
2.数据除以255，使像素值归一化至[0 - 1]之间