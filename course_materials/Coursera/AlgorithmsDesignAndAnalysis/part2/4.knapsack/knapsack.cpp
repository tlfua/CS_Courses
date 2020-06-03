#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>


enum
{
    ALGO_BUTTOM_UP = 1,
    ALGO_TOP_DOWN = 2
};


struct Item
{
    int value, weight;
    Item()
      : value(0),
        weight(0)
    {}

    Item(int value, int weight)
      : value(value),
        weight(weight)
    {}
};


class Memo
{
  public:
    Memo()
    {}

    void Put(int idx_item, int capacity, int value)
    {
        if (map_.find(idx_item) == map_.end())
            map_.insert(std::pair<int, std::unordered_map<int, int>>(
                                    idx_item, std::unordered_map<int, int>()));
        auto& fst_entry = map_.at(idx_item);
        fst_entry.insert(std::pair<int, int>(capacity, value));
    }

    int Get(int idx_item, int capacity)
    {
        if (map_.find(idx_item) == map_.end())
            return kNotRecorded;
        const auto& fst_entry = map_.at(idx_item);
        const auto& snd_entry = fst_entry.find(capacity);
        return (snd_entry != fst_entry.end())? snd_entry->second : kNotRecorded;
    }

    static constexpr const int kNotRecorded = -1;

  private:
    std::unordered_map<int, std::unordered_map<int, int>> map_;
};


void KnapsackButtomUp(std::vector<Item>& vec_item, int count_item, int capacity)
{
    std::vector<std::vector<int>> dp(count_item + 1);
    for (int i = 0 ; i <= count_item ; ++i) {
        dp[i].resize(capacity + 1);
        dp[i][0] = 0;
    }
    for (int i = 1 ; i <= capacity ; ++i)
        dp[0][i] = 0;

    for (int i = 1 ; i <= count_item ; ++i) {
        for (int j = 1 ; j <= capacity ; ++j) {
            if (j < vec_item[i].weight)
                dp[i][j] = dp[i - 1][j];
            else {
                Item& item = vec_item[i];
                dp[i][j] = std::max(dp[i - 1][j],
                                   (dp[i - 1][j - item.weight]) + item.value);
            }
        }
    }

    std::cout << dp[count_item][capacity] << std::endl;
}


int TopDownTraversal(std::vector<Item>& vec_item, int idx_item, int capacity,
                     Memo* p_memo, int* optimal)
{
    int cache = p_memo->Get(idx_item, capacity);
    if (cache != Memo::kNotRecorded)
        return cache;

    Item &item = vec_item[idx_item];
    if (idx_item == 1) {
        if (capacity >= item.weight) {
            p_memo->Put(1, capacity, item.value);
            return item.value;
        } else {
            p_memo->Put(1, capacity, 0);
            return 0;
        }
    }

    int select = 0;
    if (capacity >= item.weight)
        select = item.value + TopDownTraversal(vec_item, idx_item - 1,
                                capacity - item.weight, p_memo, optimal);

    int no_select = TopDownTraversal(vec_item, idx_item - 1, capacity,
                                    p_memo, optimal);

    int large = std::max(select, no_select);
    p_memo->Put(idx_item, capacity, large);
    *optimal = (large >= *optimal)? large : *optimal;

    return large;
}

void KnapsackTopDown(std::vector<Item>& vec_item, int count_item, int capacity)
{
    int optimal = 0;
    Memo memo;
    TopDownTraversal(vec_item, count_item, capacity, &memo, &optimal);
    std::cout << optimal << std::endl;
}


int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Please specify the input file and algorithm." << std::endl;
        return -1;
    }

    std::fstream stream(argv[1]);
    int capacity, count_item;
    stream >> capacity >> count_item;

    // Read all the items.
    std::vector<Item> vec_item(1);
    for (int i = 0 ; i < count_item ; ++i) {
        int value, weight;
        stream >> value >> weight;
        vec_item.push_back(Item(value, weight));
    }

    int algo = atoi(argv[2]);
    if (algo == ALGO_BUTTOM_UP)
        KnapsackButtomUp(vec_item, count_item, capacity);
    else if (algo == ALGO_TOP_DOWN)
        KnapsackTopDown(vec_item, count_item, capacity);

    return 0 ;
}