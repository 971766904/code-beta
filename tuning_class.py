# @Time : 2022/5/16 9:56 
# @Author : zhongyu 
# @File : tuning_class.py
import lightgbm as lgb
import pandas as pd
import numpy as np
import test_analysis_shot as tas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


class Circle(object):
    pi = 3.14  # 类属性

    def __init__(self, r):
        self.r = r  # 实例属性

    def get_area(self):
        """ 圆的面积 """
        # return self.r**2 * Circle.pi # 通过实例修改pi的值对面积无影响，这个pi为类属性的值
        return self.r ** 2 * self.pi  # 通过实例修改pi的值对面积我们圆的面积就会改变


circle1 = Circle(1)
print(circle1.get_area())
