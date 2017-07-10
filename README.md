# captcha
recognition of fixed length captcha

Requirement:
tensorflow 1.1 and slim


知乎用户项亮的一篇文章讲了在mxnet下使用cnn实现定长和不定长验证码的识别，本文尝试在tensorflow实现定长验证码的识别：https://zhuanlan.zhihu.com/p/21344595

程序随机生成的验证码：
![随机生成的验证码](https://github.com/lsqpku/captcha/raw/master/captcha.jpg)

随着持续训练，loss逐渐降低，精度逐渐提高，4位长度的验证码，京都可以超过98%：
![tensorboard](https://github.com/lsqpku/captcha/raw/master/captcha_tensorboard.jpg)
