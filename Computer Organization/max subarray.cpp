#include<iostream>
using namespace std;

int maxSubarray(int nums[]) {
	int begin, end = 0;
	int maxSum = nums[0];
	int curSum = nums[0];
	for (int i = 1; i < 16; i++) {
		curSum = max(nums[i], curSum + nums[i]);//要不要包含前面
		maxSum = max(maxSum, curSum);		 	//目前最大的和
		if ( maxSum == nums[i] ) begin = i;
		if ( maxSum == curSum ) end = i;
  	}
  	cout << "從第" << begin << "個到" << end << "個\n";
	return maxSum;
}

int main(){
	//16個，採二補數，記憶體一格19bit，從左開始19符號，18~6進位備用，5~0為數字
	int list[] = {-1, 2, -20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	cout << maxSubarray(list);
	return 0;
}




