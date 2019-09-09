# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:17:45 2019

@author: user
"""
# indent 주의
def add1(a, b, c=5):
    z1 = a*2
    z2 = a+b
    z3 = a+c
    # 리스트, 튜플, 딕셔너리 등 자유롭게 리턴 가능
    return [z1, z2, z3]

def add2(a, b):
    z1 = a*b
    z2 = (a+b)/2
    return [z1, z2]