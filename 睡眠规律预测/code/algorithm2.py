# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 20:08
# @Author  : XiaoMa（小马）
# @qq      : 1530253396（任何问题欢迎联系）
# @File    : algorithm2.py
import time
import functools
def test(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('耗时：',(end - start)*1000,'ms')
    return wrapper
# @test
def GCD(x, y):
    return x if y == 0 else GCD(y,x%y)
# @test
def ExGCD(a, b):
    """
    a: 模的取值
    b: 想求逆的值
    """
    if (b == 0):
        return 1, 0, a
    x, y, gcd = ExGCD(b, a % b)
    return y, x-a//b*y, gcd
def algorithm2():
    param = [(7,5),
             (31,-13),
             (24,36),
             (2461502723515673086658704256944912426065172925575,1720876577542770214811199308823476528929542231719),
             (13709616469144948883512229123502305176385931810284088906755090238431898972708904439178898468021710798401875986657125211084472621499595371254346390738382042,19235039994987625167590963480899777255933775238312044097122773255647530276806317636026727679800825370459321617724871515442147432420951257037823141069640181),
             (96557807278640299121519463045206377934978887298086994211590515571717325955785923783159432436307870512742354877476790046891802153053719263845602618422474671707896136814707875793300040916757228826108499490311295942553478010913043680523612655400526255290702983490382191419067057726624348815391509161304477322782,146116799305702219220540123503890666704710410600856387071776221592477256752759997798169931809156426471243799795374072510423645363680537337813774268658907130969994146783451692837222772144941434909050652825715582967684984814095461041109999161468223272534833391335036612863782740784573110824091866969655931097032)
             ]
    for it in param:
        i,j = it[0], it[1]
        s1 = time.time()
        x = GCD(i, j)
        s2 = time.time()
        a, b, c = ExGCD(i, j)
        s3 = time.time()
        print(f'样例{i} {j}的\nGCD的结果为:\n{x}\nEXGCD的结果为:\n{a}, {b}, {c}')
if __name__ == "__main__":
    algorithm2()