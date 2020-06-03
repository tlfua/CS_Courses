#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<vector>

using namespace std;

#define SIZE 100
char line[SIZE];

void merge(int *arr, int size1, int size2, long *inversions) ;
void mergeSort(int *arr, int size, long* inversions) ;

int main ()
{
    int A[100000];

    vector<int> V;
    int i_temp;
    int ii = 0;

    fstream fin;

    fin.open("IntegerArray.txt", ios::in);
    //fin.open("test1.txt", ios::in);

    while( fin.getline(line, sizeof(line), '\n') != NULL ){
        i_temp = atoi(line);
        V.push_back(i_temp);
        A[ii] = i_temp;
        ii++;
    }

    cout<<"A: \n";
    cout<<ii<<"\n";
    cout << A[ii - 1] << "\n";

    long Invs = 0;
    mergeSort(A, ii, &Invs);

    cout<<"Invs = "<<Invs<<"\n";

    fin.close();
    return 0;
}

void merge(int *arr, int size1, int size2, long *inversions) {
    int temp[size1+size2];
    int ptr1=0, ptr2=0;

    while (ptr1+ptr2 < size1+size2) {
        if (ptr1 < size1 && arr[ptr1] <= arr[size1+ptr2] || ptr1 < size1 && ptr2 >= size2)
            temp[ptr1+ptr2] = arr[ptr1++];

        if (ptr2 < size2 && arr[size1+ptr2] < arr[ptr1] || ptr2 < size2 && ptr1 >= size1) {
            temp[ptr1+ptr2] = arr[size1+ptr2++];
            *inversions += size1-ptr1;
        }
    }

    for (int i=0; i < size1+size2; i++)
        arr[i] = temp[i];
}

void mergeSort(int *arr, int size, long* inversions) {
    if (size == 1)
        return;

    int size1 = size/2, size2 = size-size1;
    mergeSort(arr, size1, inversions);
    mergeSort(arr+size1, size2, inversions);
    merge(arr, size1, size2, inversions);
}