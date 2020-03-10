# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 20:02
# @Author  : XiaoMa（小马）
# @qq      : 1530253396（任何问题欢迎联系）
# @File    : algorithm5.py
import random


def fast_power(base, power, n):
    result = 1
    tmp = base
    while power > 0:
        if power & 1 == 1:
            result = (result * tmp) % n
        tmp = (tmp * tmp) % n
        power = power >> 1
    return result


def Miller_Rabin(n, iter_num):
    # 2 is prime
    if n == 2:
        return True
    # if n is even or less than 2, then n is not a prime
    if n & 1 == 0 or n < 2:
        return False
    # n-1 = (2^s)m
    m, s = n - 1, 0
    while m & 1 == 0:
        m = m >> 1
        s += 1
    # M-R test
    for _ in range(iter_num):
        b = fast_power(random.randint(2, n - 1), m, n)
        if b == 1 or b == n - 1:
            continue
        for __ in range(s - 1):
            b = fast_power(b, 2, n)
            if b == n - 1:
                break
        else:
            return False
    return True


if __name__ == "__main__":
    # example
    nums = [1000023,1000033, 100160063, 1500450271, 1494462659429290047815067355171411187560751791530]
    for i in nums:
        flag = 1 if Miller_Rabin(i, 10) else 0
        if flag == 1:
            print(f'{i} 是素数')
        else:
            print(f'{i} 不是素数')