// C++ program for Huffman Coding
#include <iostream>
#include <fstream>
#include <sstream>
#include <bits/stdc++.h>

using namespace std;

string codeWord[1000];
int cnt_codeWord= 0;

// A Huffman tree node
struct MinHeapNode
{
    char data;                // One of the input characters
    unsigned freq;             // Frequency of the character
    MinHeapNode *left, *right; // Left and right child

    MinHeapNode(char data, unsigned freq)
    {
        left = right = NULL;
        this->data = data;
        this->freq = freq;
    }
};

// For comparison of two heap nodes (needed in min heap)
struct compare
{
    bool operator()(MinHeapNode* l, MinHeapNode* r)
    {
        return (l->freq > r->freq);
    }
};

// Prints huffman codes from the root of Huffman Tree.
void printCodes(struct MinHeapNode* root, string str)
{
    if (!root)
        return;

    if (root->data != '$') {
        //cout << root->data << ": " << str << "\n";
        codeWord[cnt_codeWord]= str;
        cnt_codeWord++;
    }

    printCodes(root->left, str + "0");
    printCodes(root->right, str + "1");
}

// The main function that builds a Huffman Tree and
// print codes by traversing the built Huffman Tree
void HuffmanCodes(char data[], int freq[], int size)
{
    struct MinHeapNode *left, *right, *top;

    // Create a min heap & inserts all characters of data[]
    priority_queue<MinHeapNode*, vector<MinHeapNode*>, compare> minHeap;
    for (int i = 0; i < size; ++i)
        minHeap.push(new MinHeapNode(data[i], freq[i]));

    // Iterate while size of heap doesn't become 1
    while (minHeap.size() != 1)
    {
        // Extract the two minimum freq items from min heap
        left = minHeap.top();
        minHeap.pop();

        right = minHeap.top();
        minHeap.pop();

        // Create a new internal node with frequency equal to the
        // sum of the two nodes frequencies. Make the two extracted
        // node as left and right children of this new node. Add
        // this node to the min heap
        // '$' is a special value for internal nodes, not used
        top = new MinHeapNode('$', left->freq + right->freq);
        top->left = left;
        top->right = right;
        minHeap.push(top);
    }

    // Print Huffman codes using the Huffman tree built above
    printCodes(minHeap.top(), "");
}

// Driver program to test above functions
template <class T>
void convertFromString(T &value, const std::string &s)
{
    std::stringstream ss(s);
    ss >> value;
}

int main()
{
    /*
    char arr[] = { 'a', 'b', 'c', 'd', 'e', 'f' };
    int freq[] = { 5, 9, 12, 13, 16, 45 };
    int size = sizeof(arr) / sizeof(arr[0]);
    */
    //HuffmanCodes(arr, freq, size);

    char arr[1000];
    int freq[1000];
    int size= 1000;

    std::ifstream iFile;
    iFile.open("huffman.txt", std::ifstream::in);

    int tmpW;
    string line;

    getline(iFile, line); // get first line which is size

    // get every symbol weight
    int cnt= 0;
    while (getline(iFile, line)) {
        convertFromString(tmpW, line);
        freq[cnt]= tmpW;
        cnt++;
    }
    iFile.close();

    // assign symbol
    for (int i= 0; i < 1000; i++) {
        arr[i]= 'a';
    }
    HuffmanCodes(arr, freq, size);

    int max_size= codeWord[0].size();
    int min_size= codeWord[0].size();

    for (int i= 0; i < 1000; i++) {
        if ( codeWord[i].size() > max_size )
            max_size= codeWord[i].size();
        if ( codeWord[i].size() < min_size )
            min_size= codeWord[i].size();
    }

    std::cout << "max_size= " << max_size << ", min_size= " << min_size << "\n";
    return 0;
}