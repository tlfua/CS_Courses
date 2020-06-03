#include <iostream>
#include <fstream>
#include <sstream>

long arrW[1001];
long arrMW[1001];
bool checkMWSet[1001];
#define headIndex 1
#define tailIndex 1000

template <class T>
void convertFromString(T &value, const std::string &s)
{
    std::stringstream ss(s);
    ss >> value;
}

int main()
{
    std::ifstream iFile;
    iFile.open("mwis.txt", std::ifstream::in);

    int tmpW;
    std::string line;

    getline(iFile, line); // get first line which is size

    // get every symbol weight
    arrW[0]= 0;
    int cnt= 1;
    while (getline(iFile, line)) {
        convertFromString(tmpW, line);
        arrW[cnt]= tmpW;
        cnt++;
    }
    iFile.close();

    // Dynamic Programing
    arrMW[0]= 0;
    arrMW[1]= arrW[1];

    for (int i= headIndex + 1; i <= tailIndex; i++) {
        arrMW[i-1] > (arrMW[i-2] + arrW[i]) ? arrMW[i]= arrMW[i-1] : arrMW[i]= arrMW[i-2] + arrW[i];
    }

    // Re-construction
    for (int i= 0; i <= tailIndex; i++) {
        checkMWSet[i]= 0;
    }

    int idx= tailIndex;
    while ( idx >= headIndex ) {
        if ( arrMW[idx - 1] > arrMW[idx - 2] + arrW[idx] ) {
            idx--;
        } else {
            checkMWSet[idx]= 1;
            idx= idx - 2;
        }
    }
    // print answer
    std::cout << checkMWSet[1]
                    << checkMWSet[2]
                    << checkMWSet[3]
                    << checkMWSet[4]
                    << checkMWSet[17]
                    << checkMWSet[117]
                    << checkMWSet[517]
                    << checkMWSet[997] << "\n";
    return 0;
}