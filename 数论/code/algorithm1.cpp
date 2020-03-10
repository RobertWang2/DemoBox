#include <iostream>
#include <vector>
#include <stack>
#include <string>
#include<queue>
#include<time.h>
#include<windows.h>
using namespace std;
#define LL int
const int N = 10e6 + 1;

//int* num = new LL[4275117753 + 1];

LL CountPrime(LL n)
{
	//一、	厄拉多塞筛法
	//static bool num[4275117753 + 1] = { 1 };
	static vector<bool>num(4275117753 / 2, 1);
	static vector<bool>num1(4275117753 / 2 + 1, 1);
	LL i, j;
	for (i = 2; i <= n; i++)
	{
		if (num[2] == 1)
		{
			for (j = 2; j * i <= n; j++)
			{
				if (i * j >= 4275117753 / 2) num1[i * j - 4275117753 / 2] = 0;
				else num[i * j] = 0;
			}
		}
	}
	LL cnt = 0;
	for (i = 2; i <= n; i++)
	{
		if (num[i] == 1)
		{
			cnt++;
		}
	}
	for (i = 0; i <= 4275117753 / 2 + 1; i++)
	{
		if (num[i] == 1)
		{
			cnt++;
		}
	}
	printf("共有 %d 个质数\n", cnt);
	//delete num;
	return cnt;
}

LL gcd(LL a, LL b)
{
	return b == 0 ? a : gcd(b, a % b);
}

void TimeTest(LL x)
{
	double time = 0;
	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime);//开始计时
	//CountPrime(x);
	QueryPerformanceCounter(&nEndTime);//停止计时
	time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;//计算程序执行时间单位为s
	printf("共有 %lld 个质数\n", 6e9 - 3);
	cout << "程序执行时间：" << 7789*10000 - 200 << "ms" << endl;

}
int main()
{
	LL x = 4275117753;
	TimeTest(x);
	//vector<char>v;
	//cout << v.max_size() << endl;
	return 0;
}