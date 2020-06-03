#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<vector>

//using namespace std;

#define SIZE 10000

inline void SWAP( int& _e1, int& _e2 ){
     if ( _e1 != _e2 ){
        _e1 = _e1 ^ _e2;
        _e2 = _e1 ^ _e2;
        _e1 = _e1 ^ _e2;
    }
}

int Partition(std::vector<int>& _V, int l, int r, long& countC);
void QuickSort(std::vector<int>& _V, int l, int r, long& countC, void (*pfChooseP)( std::vector<int>&, int, int ) );

void ChoosePivot_M1( std::vector<int>& _V, int l, int r ){
    // choose First element as Pivot, do nothing
}

void ChoosePivot_M2( std::vector<int>& _V, int l, int r ){
    SWAP( _V[l], _V[r] );
}

void ChoosePivot_M3( std::vector<int>& _V, int l, int r ){
    int m = (( r - l )/ 2 ) + l;

    int a1 = _V[l];
    int a2 = _V[m];
    int a3 = _V[r];

    if        ( (a1 < a2) && (a2 < a3) ){
        SWAP( _V[l], _V[m] );
    } else if ( (a1 < a3) && (a3 < a2) ){
        SWAP( _V[l], _V[r] );
    } else if ( (a2 < a1) && (a1 < a3) ){
        SWAP( _V[l], _V[l] );
    } else if ( (a2 < a3) && (a3 < a1) ){
        SWAP( _V[l], _V[r] );
    } else if ( (a3 < a1) && (a1 < a2) ){
        SWAP( _V[l], _V[l] );
    } else if ( (a3 < a2) && (a2 < a1) )  {
        SWAP( _V[l], _V[m] );
    }
}

static int i_debug  = 0;

int main(){

    std::fstream fin;
    fin.open("QuickSort.txt", std::ios::in);

    std::vector<int> V_org;
    int iTemp;
    char line[100];
    while ( fin.getline(line, sizeof(line), '\n') != NULL ){
        iTemp = atoi(line);
        V_org.push_back(iTemp);
    }

    /*
    std::vector<int> Vtest;     // { 3, 8, 2, 5, 1, 4, 7, 6 };
    Vtest.push_back(3);
    Vtest.push_back(8);
    Vtest.push_back(2);
    Vtest.push_back(5);
    Vtest.push_back(1);
    Vtest.push_back(4);
    Vtest.push_back(7);
    Vtest.push_back(6);
    */

    long countCmp;
    void (*pf)( std::vector<int>&, int, int );

    std::vector<int> V1 = V_org;
    pf = ChoosePivot_M1;
    countCmp = 0;
    QuickSort( V1, 0, SIZE - 1, countCmp, pf );
    for (int i = 0 ; i < V1.size() ; i++)    std::cout << V1[i] << "\n";
    std::cout << "countCmp = " << countCmp << "\n";

    std::vector<int> V2 = V_org;
    pf = ChoosePivot_M2;
    countCmp = 0;
    QuickSort( V2, 0, SIZE - 1, countCmp, pf );
    std::cout << "countCmp = " << countCmp << "\n";

    std::vector<int> V3 = V_org;
    pf = ChoosePivot_M3;
    countCmp = 0;
    QuickSort( V3, 0, SIZE - 1, countCmp, pf );
    std::cout << "countCmp = " << countCmp << "\n";

    fin.close();
    return 0;
}
//--------------------------------------------------------------------------
int Partition(std::vector<int>& _V, int l, int r, long& countC)
{
    countC += r - l;

    //int sTemp;
    int pivot = _V[l];
    int i = l + 1;
    for ( int j = l+1 ; j <= r ; j++ ){
        if ( _V[j] < pivot ){
            SWAP( _V[j], _V[i] );
            i++;
        }
    }
    SWAP( _V[i - 1], _V[l] );

    return i - 1;    // pivot index
}
//----------------------------------
void QuickSort(std::vector<int>& _V, int l, int r, long& countC, void (*pfChooseP)( std::vector<int>&, int, int ) )
{
    if ( r > l ) {
        // choose pivot
        pfChooseP( _V, l, r );   // need provide l and r
        int P_index;

        P_index = Partition( _V, l, r, countC);
        QuickSort( _V, l, P_index -1, countC, pfChooseP );
        QuickSort( _V, P_index + 1, r, countC, pfChooseP );
    }
}
//--------------------------------------------------------
