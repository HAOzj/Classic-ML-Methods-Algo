# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on 29 FEB, 2020

@author: woshihaozhaojun@sina.com
"""
import math
import numpy as np
import matplotlib.pyplot as plt

RATIO = 3  # 椭圆的长宽比
LIMIT = 1.2  # 图像的坐标轴范围


class PlotComparaison(object):
    """多种优化器来优化函数 x1^2 + x2^2 * RATIO^2.

    每次参数改变为(d1, d2).梯度为(dx1, dx2)

    t+1次迭代,
    标准GD,
        d1_{t+1} = - eta * dx1
        d2_{t+1} = - eta * dx2

    带Nesterov Momentum,
        d1_{t+1} = eta * (mu * d1_t - dx1_{t+1})
        d2_{t+1} = eta * (mu * d2_t - dx2_{t+1})

    RMSProp,
        w1_{t+1} = beta2 * w1_t + (1 - beta2) * dx1_t^2
        w2_{t+1} = beta2 * w2_t + (1 - beta2) * dx2_t^2
        d1_{t+1} = - eta * dx1_t / (sqrt(w1_{t+1}) + epsilon)
        d2_{t+1} = - eta * dx2_t / (sqrt(w2_{t+1}) + epsilon)

    Adam,每次参数改变为(d1, d2)
        v1_{t+1} = beta1 * v1_t + (1 - beta1) * dx1_t
        v2_{t+1} = beta1 * v2_t + (1 - beta1) * dx2_t
        w1_{t+1} = beta2 * w1_t + (1 - beta2) * dx1_t^2
        w2_{t+1} = beta2 * w2_t + (1 - beta2) * dx2_t^2

        v1_corrected = v1_{t+1} / (1 - beta1^{t+1})
        v2_corrected = v2_{t+1} / (1 - beta1^{t+1})
        w1_corrected = w1_{t+1} / (1 - beta2^{t+1})
        w2_corrected = w2_{t+1} / (1 - beta2^{t+1})
        d1_{t+1} = - eta * v1_corrected / (sqrt(w1_corrected) + epsilon)
        d2_{t+1} = - eta * v2_corrected / (sqrt(w2_corrected) + epsilon)

    AdaGrad,
        w1_{t+1} = w1_t + dx1_t^2
        w2_{t+1} = w2_t + dx2_t^2
        d1_{t+1} = - eta * dx1_t / sqrt(w1_{t+1} + epsilon)
        d2_{t+1} = - eta * dx2_t / sqrt(w2_{t+1} + epsilon)

    Adadelta
        update1_{t+1} = rho * update1_t + (1 - rho) * d1_t^2
        update2_{t+1} = rho * update2_t + (1 - rho) * d2_t^2
        w1_{t+1} = rho * w1_t + (1 - rho) * dx1_t^2
        w2_{t+1} = rho * w2_t + (1 - rho) * dx2_t^2

        d1_{t+1} = - dx1 * rms(update1_{t+1}) / rms(w1_{t+1})
        d2_{t+1} = - dx2 * rms(update2_{t+1}) / rms(w2_{t+1})

        定义 rms(x) = sqrt(x + epsilon)

    """

    def __init__(self, eta=0.1, mu=0.9, beta1=0.9, beta2=0.99, rho=0.9, epsilon=1e-10, angles=None, contour_values=None,
                 stop_condition=1e-4):
        # 全部算法的学习率
        self.eta = eta

        # 启发式学习的终止条件
        self.stop_condition = stop_condition

        # Nesterov Momentum超参数
        self.mu = mu

        # RMSProp超参数
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Adadelta的超参数
        self.rho = rho

        # 用正态分布随机生成初始点
        self.x1_init, self.x2_init = np.random.uniform(LIMIT / 2, LIMIT), np.random.uniform(LIMIT / 2, LIMIT) / RATIO
        self.x1, self.x2 = self.x1_init, self.x2_init

        # 等高线相关
        if angles == None:
            angles = np.arange(0, 2 * math.pi, 0.01)
        self.angles = angles
        if contour_values == None:
            contour_values = [0.25 * i for i in range(1, 5)]
        self.contour_values = contour_values
        setattr(self, "contour_colors", None)

    def draw_common(self, title):
        """画等高线,最优点和设置图片各种属性"""
        # 坐标轴尺度一致
        plt.gca().set_aspect(1)

        # 根据等高线的值生成坐标和颜色
        # 海拔越高颜色越深
        num_contour = len(self.contour_values)
        if not self.contour_colors:
            self.contour_colors = [(i / num_contour, i / num_contour, i / num_contour) for i in range(num_contour)]
            self.contour_colors.reverse()
            self.contours = [
                [
                    list(map(lambda x: math.sin(x) * math.sqrt(val), self.angles)),
                    list(map(lambda x: math.cos(x) * math.sqrt(val) / RATIO, self.angles))
                ]
                for val in self.contour_values
            ]

        # 画等高线
        for i in range(num_contour):
            plt.plot(self.contours[i][0],
                     self.contours[i][1],
                     linewidth=1,
                     linestyle='-',
                     color=self.contour_colors[i],
                     label="y={}".format(round(self.contour_values[i], 2))
                     )

        # 画最优点
        plt.text(0, 0, 'x*')

        # 图片标题
        plt.title(title)

        # 设置坐标轴名字和范围
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim((-LIMIT, LIMIT))
        plt.ylim((-LIMIT, LIMIT))

        # 显示图例
        plt.legend(loc=1)

    def forward_gd(self):
        """SGD一次迭代"""
        self.d1 = -self.eta * self.dx1
        self.d2 = -self.eta * self.dx2
        self.ite += 1

    def draw_gd(self, num_ite=5):
        """画基础SGD的迭代优化.

        包括每次迭代的点,以及表示每次迭代改变的箭头
        """
        # 初始化
        setattr(self, "ite", 0)
        setattr(self, "x1", self.x1_init)
        setattr(self, "x2", self.x2_init)

        # 画每次迭代
        self.point_colors = [(i / num_ite, 0, 0) for i in range(num_ite)]
        plt.scatter(self.x1, self.x2, color=self.point_colors[0])
        for _ in range(num_ite):
            self.forward_gd()

            # 迭代的箭头
            plt.arrow(self.x1, self.x2, self.d1, self.d2,
                      length_includes_head=True,
                      linestyle=':',
                      label='{} ite'.format(self.ite),
                      color='b',
                      head_width=0.08
                      )

            self.x1 += self.d1
            self.x2 += self.d2
            print("第{}次迭代后,坐标为({}, {})".format(self.ite, self.x1, self.x2))
            plt.scatter(self.x1, self.x2)  # 迭代的点

    def forward_momentum(self):
        """带Momentum的SGD一次迭代"""
        self.d1 = self.eta * (self.mu * self.d1_pre - self.dx1)
        self.d2 = self.eta * (self.mu * self.d2_pre - self.dx2)
        self.ite += 1
        self.d1_pre, self.d2_pre = self.d1, self.d2

    def draw_momentum(self, num_ite=5):
        """画带Momentum的迭代优化."""
        # 初始化
        setattr(self, "ite", 0)
        setattr(self, "x1", self.x1_init)
        setattr(self, "x2", self.x2_init)
        setattr(self, "d1_pre", 0)
        setattr(self, "d2_pre", 0)

        # 画每次迭代
        self.point_colors = [(i / num_ite, 0, 0) for i in range(num_ite)]
        plt.scatter(self.x1, self.x2, color=self.point_colors[0])
        for _ in range(num_ite):
            self.forward_momentum()
            # 迭代的箭头
            plt.arrow(self.x1, self.x2, self.d1, self.d2,
                      length_includes_head=True,
                      linestyle=':',
                      label='{} ite'.format(self.ite),
                      color='b',
                      head_width=0.08
                      )

            self.x1 += self.d1
            self.x2 += self.d2
            print("第{}次迭代后,坐标为({}, {})".format(self.ite, self.x1, self.x2))
            plt.scatter(self.x1, self.x2)  # 迭代的点
            if self.loss < self.stop_condition:
                break

    def forward_rmsprop(self):
        """RMSProp一次迭代"""
        w1 = self.beta2 * self.w1_pre + (1 - self.beta2) * (self.dx1 ** 2)
        w2 = self.beta2 * self.w2_pre + (1 - self.beta2) * (self.dx2 ** 2)
        self.ite += 1
        self.w1_pre, self.w2_pre = w1, w2

        self.d1 = -self.eta * self.dx1 / (math.sqrt(w1) + self.epsilon)
        self.d2 = -self.eta * self.dx2 / (math.sqrt(w2) + self.epsilon)

    def draw_rmsprop(self, num_ite=5):
        """画RMSProp的迭代优化."""
        # 初始化
        setattr(self, "ite", 0)
        setattr(self, "x1", self.x1_init)
        setattr(self, "x2", self.x2_init)
        setattr(self, "w1_pre", 0)
        setattr(self, "w2_pre", 0)

        # 画每次迭代
        self.point_colors = [(i / num_ite, 0, 0) for i in range(num_ite)]
        plt.scatter(self.x1, self.x2, color=self.point_colors[0])
        for _ in range(num_ite):
            self.forward_rmsprop()

            # 迭代的箭头
            plt.arrow(self.x1, self.x2, self.d1, self.d2,
                      length_includes_head=True,
                      linestyle=':',
                      label='{} ite'.format(self.ite),
                      color='b',
                      head_width=0.08
                      )

            self.x1 += self.d1
            self.x2 += self.d2

            print("第{}次迭代后,坐标为({}, {})".format(self.ite, self.x1, self.x2))
            plt.scatter(self.x1, self.x2)  # 迭代的点

            if self.loss < self.stop_condition:
                break

    def forward_adam(self):
        """AdaM一次迭代"""
        w1 = self.beta2 * self.w1_pre + (1 - self.beta2) * (self.dx1 ** 2)
        w2 = self.beta2 * self.w2_pre + (1 - self.beta2) * (self.dx2 ** 2)
        v1 = self.beta1 * self.v1_pre + (1 - self.beta1) * self.dx1
        v2 = self.beta1 * self.v2_pre + (1 - self.beta1) * self.dx2
        self.ite += 1
        self.v1_pre, self.v2_pre = v1, v2
        self.w1_pre, self.w2_pre = w1, w2

        v1_corr = v1 / (1 - math.pow(self.beta1, self.ite))
        v2_corr = v2 / (1 - math.pow(self.beta1, self.ite))
        w1_corr = w1 / (1 - math.pow(self.beta2, self.ite))
        w2_corr = w2 / (1 - math.pow(self.beta2, self.ite))

        self.d1 = -self.eta * v1_corr / (math.sqrt(w1_corr) + self.epsilon)
        self.d2 = -self.eta * v2_corr / (math.sqrt(w2_corr) + self.epsilon)

    def draw_adam(self, num_ite=5):
        """画AdaM的迭代优化."""
        # 初始化
        setattr(self, "ite", 0)
        setattr(self, "x1", self.x1_init)
        setattr(self, "x2", self.x2_init)
        setattr(self, "w1_pre", 0)
        setattr(self, "w2_pre", 0)
        setattr(self, "v1_pre", 0)
        setattr(self, "v2_pre", 0)

        # 画每次迭代
        self.point_colors = [(i / num_ite, 0, 0) for i in range(num_ite)]
        plt.scatter(self.x1, self.x2, color=self.point_colors[0])
        for _ in range(num_ite):
            self.forward_adam()

            # 迭代的箭头
            plt.arrow(self.x1, self.x2, self.d1, self.d2,
                      length_includes_head=True,
                      linestyle=':',
                      label='{} ite'.format(self.ite),
                      color='b',
                      head_width=0.08
                      )

            self.x1 += self.d1
            self.x2 += self.d2

            print("第{}次迭代后,坐标为({}, {})".format(self.ite, self.x1, self.x2))
            plt.scatter(self.x1, self.x2)  # 迭代的点

            if self.loss < self.stop_condition:
                break

    def forward_adagrad(self):
        """AdaGrad一次迭代"""
        w1 = self.w1_pre + self.dx1 ** 2
        w2 = self.w2_pre + self.dx2 ** 2
        self.ite += 1
        self.w1_pre, self.w2_pre = w1, w2
        self.d1 = -self.eta * self.dx1 / math.sqrt(w1 + self.epsilon)
        self.d2 = -self.eta * self.dx2 / math.sqrt(w2 + self.epsilon)

    def draw_adagrad(self, num_ite=5):
        """画AdaGrad的迭代优化."""
        # 初始化
        setattr(self, "ite", 0)
        setattr(self, "x1", self.x1_init)
        setattr(self, "x2", self.x2_init)
        setattr(self, "w1_pre", 0)
        setattr(self, "w2_pre", 0)

        # 画每次迭代
        self.point_colors = [(i / num_ite, 0, 0) for i in range(num_ite)]
        plt.scatter(self.x1, self.x2, color=self.point_colors[0])
        for _ in range(num_ite):
            self.forward_adagrad()

            # 迭代的箭头
            plt.arrow(self.x1, self.x2, self.d1, self.d2,
                      length_includes_head=True,
                      linestyle=':',
                      label='{} ite'.format(self.ite),
                      color='b',
                      head_width=0.08
                      )

            self.x1 += self.d1
            self.x2 += self.d2

            print("第{}次迭代后,坐标为({}, {})".format(self.ite, self.x1, self.x2))
            plt.scatter(self.x1, self.x2)  # 迭代的点

            if self.loss < self.stop_condition:
                break

    def forward_adadelta(self):
        """Adadelta一次迭代"""
        w1 = self.rho * self.w1_pre + (1 - self.rho) * (self.dx1 ** 2)
        w2 = self.rho * self.w2_pre + (1 - self.rho) * (self.dx2 ** 2)
        update1 = self.rho * self.update1_pre + (1 - self.rho) * (self.d1 ** 2)
        update2 = self.rho * self.update2_pre + (1 - self.rho) * (self.d2 ** 2)
        self.ite += 1
        self.update1_pre, self.update2_pre = update1, update2
        self.w1_pre, self.w2_pre = w1, w2

        self.d1 = - self.rms(update1) / self.rms(w1) * self.dx1
        self.d2 = - self.rms(update2) / self.rms(w2) * self.dx2

    def draw_adadelta(self, num_ite=5):
        """画Adadelta的迭代优化."""
        # 初始化
        for attr in ["w{}_pre", "update{}_pre", "d{}"]:
            for dim in [1, 2]:
                setattr(self, attr.format(dim), 0)
        setattr(self, "ite", 0)
        setattr(self, "x1", self.x1_init)
        setattr(self, "x2", self.x2_init)

        # 画每次迭代
        self.point_colors = [(i / num_ite, 0, 0) for i in range(num_ite)]
        plt.scatter(self.x1, self.x2, color=self.point_colors[0])
        for _ in range(num_ite):
            self.forward_adadelta()

            # 迭代的箭头
            plt.arrow(self.x1, self.x2, self.d1, self.d2,
                      length_includes_head=True,
                      linestyle=':',
                      label='{} ite'.format(self.ite),
                      color='b',
                      head_width=0.08
                      )

            self.x1 += self.d1
            self.x2 += self.d2

            print("第{}次迭代后,坐标为({}, {})".format(self.ite, self.x1, self.x2))
            plt.scatter(self.x1, self.x2)  # 迭代的点

            if self.loss < self.stop_condition:
                break

    @property
    def dx1(self):
        return self.x1 * 2

    @property
    def dx2(self):
        return self.x2 * 2 * (RATIO ** 2)

    @property
    def loss(self):
        return self.x1 ** 2 + (RATIO * self.x2) ** 2

    def rms(self, x):
        return math.sqrt(x + self.epsilon)

    def show(self):
        # 设置图片大小
        plt.figure(figsize=(20, 20))
        # 展示
        plt.show()

