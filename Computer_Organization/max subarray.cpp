#include<iostream>
using namespace std;

int maxSubarray(int nums[]) {
	int begin, end = 0;
	int maxSum = nums[0];
	int curSum = nums[0];
	for (int i = 1; i < 16; i++) {
		curSum = max(nums[i], curSum + nums[i]);//�n���n�]�t�e��
		maxSum = max(maxSum, curSum);		 	//�ثe�̤j���M
		if ( maxSum == nums[i] ) begin = i;
		if ( maxSum == curSum ) end = i;
  	}
  	cout << "�q��" << begin << "�Ө�" << end << "��\n";
	return maxSum;
}

int main(){
	//16�ӡA�ĤG�ɼơA�O����@��19bit�A�q���}�l19�Ÿ��A18~6�i��ƥΡA5~0���Ʀr
	int list[] = {-1, 2, -20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	cout << maxSubarray(list);
	return 0;
}




