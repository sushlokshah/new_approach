#include <iostream>
#include <vector>

int searchInSorted(int arr[], int N, int K) 
{ 

    // Your code here
    int start = 0, end = N-1;
    int min = (end - start)/2;
    
    while(start > end)
    {
        
        std::cout << arr[min] << std::endl;
        if(arr[min] > K)
        {
            end = min - 1;
            min = (end - start)/2;
        //   cout<< min<< endl;
        }
        else
        {
            if(arr[min] < K)
            {
                start = min + 1;
                min = (end - start)/2;
            //   cout << min << endl;
            
            }
            else{
                if(arr[min] == K)
                {
                    return 1;
                }
            }  
        }
        
    }
    return -1;
}

int main() 
{ 
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; 
    int N = sizeof(arr)/sizeof(arr[0]); 
    int K = 5; 
    int result = searchInSorted(arr, N, K); 
    if (result == 1) 
        std::cout << "Found"; 
    else
        std::cout << "Not Found"; 
    return 0; 
}