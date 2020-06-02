#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <assert.h>

/** The key value pair for associative data structures. */
typedef struct _Pair {
    void* key;
    void* value;
} Pair;

typedef struct _VectorData VectorData;
typedef int (*VectorCompare) (const void*, const void*);
typedef void (*VectorClean) (void*);

typedef struct _Vector {
    VectorData *data;
    bool (*push_back) (struct _Vector*, void*);
    bool (*insert) (struct _Vector*, unsigned, void*);
    bool (*pop_back) (struct _Vector*);
    bool (*remove) (struct _Vector*, unsigned);
    bool (*set) (struct _Vector*, unsigned, void*);
    bool (*get) (struct _Vector*, unsigned, void**);
    bool (*resize) (struct _Vector*, unsigned);
    unsigned (*size) (struct _Vector*);
    unsigned (*capacity) (struct _Vector*);
    void (*sort) (struct _Vector*, VectorCompare);
    void (*first) (struct _Vector*, bool);
    bool (*next) (struct _Vector*, void**);
    bool (*reverse_next) (struct _Vector*, void**);
    void (*set_clean) (struct _Vector*, VectorClean);
} Vector;

Vector* VectorInit(unsigned cap);
void VectorDeinit(Vector* obj);
bool VectorPushBack(Vector* self, void* element);
bool VectorInsert(Vector* self, unsigned idx, void* element);
bool VectorPopBack(Vector* self);
bool VectorRemove(Vector* self, unsigned idx);
bool VectorSet(Vector* self, unsigned idx, void* element);
bool VectorGet(Vector* self, unsigned idx, void** p_element);
bool VectorResize(Vector* self, unsigned capacity);
unsigned VectorSize(Vector* self);
unsigned VectorCapacity(Vector* self);
void VectorSort(Vector* self, VectorCompare func);
void VectorFirst(Vector* self, bool is_reverse);
bool VectorNext(Vector* self, void** p_element);
bool VectorReverseNext(Vector* self, void** p_element);
void VectorSetClean(Vector* self, VectorClean func);

struct _VectorData {
    unsigned size_;
    unsigned capacity_;
    unsigned iter_;
    void** elements_;
    VectorClean func_clean_;
};

/*===========================================================================*
 *                  Definition for internal operations                       *
 *===========================================================================*/
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

bool _VectorReisze(VectorData* data, unsigned capacity);


/*===========================================================================*
 *               Implementation for the exported operations                  *
 *===========================================================================*/
Vector* VectorInit(unsigned capacity)
{
    Vector* obj = (Vector*)malloc(sizeof(Vector));
    if (unlikely(!obj))
        return NULL;

    VectorData* data = (VectorData*)malloc(sizeof(VectorData));
    if (unlikely(!data)) {
        free(obj);
        return NULL;
    }

    void** elements = (void**)malloc(sizeof(void*) * capacity);
    if (unlikely(!elements)) {
        free(data);
        free(obj);
        return NULL;
    }

    data->size_ = 0;
    data->capacity_ = capacity;
    data->elements_ = elements;
    data->func_clean_ = NULL;

    obj->data = data;
    obj->push_back = VectorPushBack;
    obj->insert = VectorInsert;
    obj->pop_back = VectorPopBack;
    obj->remove = VectorRemove;
    obj->set = VectorSet;
    obj->get = VectorGet;
    obj->resize = VectorResize;
    obj->size = VectorSize;
    obj->capacity = VectorCapacity;
    obj->sort = VectorSort;
    obj->first = VectorFirst;
    obj->next = VectorNext;
    obj->reverse_next = VectorReverseNext;
    obj->set_clean = VectorSetClean;

    return obj;
}

void VectorDeinit(Vector *obj)
{
    if (unlikely(!obj))
        return;

    VectorData* data = obj->data;
    VectorClean func_clean = data->func_clean_;
    void** elements = data->elements_;
    unsigned size = data->size_;

    unsigned i;
    for (i = 0 ; i < size ; ++i) {
        if (func_clean)
            func_clean(elements[i]);
    }

    free(elements);
    free(data);
    free(obj);
    return;
}

bool VectorPushBack(Vector* self, void* element)
{
    VectorData* data = self->data;
    unsigned size = data->size_;
    unsigned capacity = data->capacity_;

    /* If the internal array is full, extend it to double capacity. */
    if (size == capacity) {
        bool rtn = _VectorReisze(data, capacity << 1);
        if (rtn == false)
            return false;
    }

    data->elements_[size] = element;
    data->size_++;
    return true;
}

bool VectorInsert(Vector* self, unsigned idx, void* element)
{
    VectorData* data = self->data;
    unsigned size = data->size_;
    unsigned capacity = data->capacity_;

    if (unlikely(idx > size))
        return false;

    /* If the internal array is full, extend it to double capacity. */
    if (size == capacity) {
        bool rtn = _VectorReisze(data, capacity << 1);
        if (rtn == false)
            return false;
    }

    /* Shift the trailing elements if necessary. */
    void** elements = data->elements_;
    unsigned num_shift = size - idx;
    if (likely(num_shift > 0))
        memmove(elements + idx + 1, elements + idx, sizeof(void*) * num_shift);

    elements[idx] = element;
    data->size_ = size + 1;
    return true;
}

bool VectorPopBack(Vector* self)
{
    VectorData* data = self->data;
    unsigned size = data->size_;
    if (unlikely(size == 0))
        return false;

    --size;
    data->size_ = size;
    void* element = data->elements_[size];
    VectorClean func_clean = data->func_clean_;
    if (func_clean)
        func_clean(element);

    return true;
}

bool VectorRemove(Vector* self, unsigned idx)
{
    VectorData* data = self->data;
    unsigned size = data->size_;
    if (unlikely(idx >= size))
        return false;

    void** elements = data->elements_;
    VectorClean func_clean = data->func_clean_;
    if (func_clean)
        func_clean(elements[idx]);

    /* Shift the trailing items if necessary. */
    unsigned num_shift = size - idx - 1;
    if (likely(num_shift > 0))
        memmove(elements + idx, elements + idx + 1, sizeof(void*) * num_shift);

    data->size_ = size - 1;
    return true;
}

bool VectorSet(Vector* self, unsigned idx, void* element)
{
    VectorData* data = self->data;
    if (unlikely(idx >= data->size_))
        return false;

    void** elements = data->elements_;
    VectorClean func_clean = data->func_clean_;
    if (func_clean)
        func_clean(elements[idx]);
    elements[idx] = element;
    return true;
}

bool VectorGet(Vector* self, unsigned idx, void** p_element)
{
    VectorData* data = self->data;
    if (unlikely(idx >= data->size_))
        return false;

    *p_element = data->elements_[idx];
    return true;
}

bool VectorResize(Vector* self, unsigned capacity)
{
    return _VectorReisze(self->data, capacity);
}

unsigned VectorSize(Vector* self)
{
    return self->data->size_;
}

unsigned VectorCapacity(Vector* self)
{
    return self->data->capacity_;
}

void VectorSort(Vector* self, VectorCompare func)
{
    VectorData* data = self->data;
    qsort(data->elements_, data->size_, sizeof(void*), func);
}

void VectorFirst(Vector* self, bool is_reverse)
{
    self->data->iter_ = (is_reverse == false)? 0 : (self->data->size_ - 1);
}

bool VectorNext(Vector* self, void** p_element)
{
    VectorData* data = self->data;
    unsigned iter = data->iter_;
    if (unlikely(iter >= data->size_))
        return false;

    *p_element = data->elements_[iter];
    data->iter_ = iter + 1;
    return true;
}

bool VectorReverseNext(Vector* self, void** p_element)
{
    VectorData* data = self->data;
    unsigned iter = data->iter_;
    if (unlikely(iter == UINT_MAX))
        return false;

    *p_element = data->elements_[iter];
    data->iter_ = iter - 1;
    return true;
}

void VectorSetClean(Vector* self, VectorClean func)
{
    self->data->func_clean_ = func;
}

/*===========================================================================*
 *               Implementation for internal operations                      *
 *===========================================================================*/
bool _VectorReisze(VectorData* data, unsigned capacity)
{
    void** elements = data->elements_;
    unsigned old_size = data->size_;

    /* Remove the trailing items if the given element number is smaller than the
       old element count. */
    if (unlikely(capacity < old_size)) {
        VectorClean func_clean = data->func_clean_;
        unsigned idx = capacity;
        while (idx < old_size) {
            if (func_clean)
                func_clean(elements[idx]);
            ++idx;
        }
        data->size_ = capacity;
    }

    if (unlikely(capacity == old_size))
        return true;

    void** new_elements = (void**)realloc(elements, sizeof(void*) * capacity);
    if (new_elements) {
        data->elements_  = new_elements;
        data->capacity_ = capacity;
    }

    return (new_elements)? true : false;
}

int CompareWord(const void* lhs, const void* rhs)
{
    char* word_lhs = *((char**)lhs);
    char* word_rhs = *((char**)rhs);
    // int i = 0;
    // while ((word_lhs[i] != '\0') && (word_rhs[i] != '\0')){
    //     if (word_lhs[i] < word_rhs[i])
    //         return -1;
    //     else if (word_lhs[i] > word_rhs[i])
    //         return 1;
    //     i++;
    // }
    // if (strlen(word_lhs) != strlen(word_rhs))
    //     return strlen(word_lhs) < strlen(word_rhs)? -1 : 1;
    // return 0;
    return strcmp(word_lhs, word_rhs);
}
///////////////////////////////////////////////////////////    End of Vector

unsigned HashMurMur32(void* key, size_t size);
unsigned HashJenkins(void* key, size_t size);
unsigned HashDjb2(char* key);

unsigned HashMurMur32(void* key, size_t size)
{
    if (!key || size == 0)
        return 0;

    const unsigned c1 = 0xcc9e2d51;
    const unsigned c2 = 0x1b873593;
    const unsigned r1 = 15;
    const unsigned r2 = 13;
    const unsigned m = 5;
    const unsigned n = 0xe6546b64;

    unsigned hash = 0xdeadbeef;

    const int nblocks = size / 4;
    const unsigned *blocks = (const unsigned*)key;
    int i;
    for (i = 0; i < nblocks; i++) {
        unsigned k = blocks[i];
        k *= c1;
        k = (k << r1) | (k >> (32 - r1));
        k *= c2;

        hash ^= k;
        hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
    }

    const uint8_t *tail = (const uint8_t*) (key + nblocks * 4);
    unsigned k1 = 0;

    switch (size & 3) {
        case 3:
            k1 ^= tail[2] << 16;
        case 2:
            k1 ^= tail[1] << 8;
        case 1:
            k1 ^= tail[0];

            k1 *= c1;
            k1 = (k1 << r1) | (k1 >> (32 - r1));
            k1 *= c2;
            hash ^= k1;
    }

    hash ^= size;
    hash ^= (hash >> 16);
    hash *= 0x85ebca6b;
    hash ^= (hash >> 13);
    hash *= 0xc2b2ae35;
    hash ^= (hash >> 16);

    return hash;
}

unsigned HashDjb2(char* key)
{
    unsigned hash = 5381;
    int c;

    while ((c = *key++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    return hash;
}

typedef struct _HashMapData HashMapData;
typedef unsigned (*HashMapHash) (void*);
typedef int (*HashMapCompare) (void*, void*);
typedef void (*HashMapCleanKey) (void*);
typedef void (*HashMapCleanValue) (void*);
typedef struct _HashMap {
    HashMapData *data;
    bool (*put) (struct _HashMap*, void*, void*);
    void* (*get) (struct _HashMap*, void*);
    bool (*contain) (struct _HashMap*, void*);
    bool (*remove) (struct _HashMap*, void*);
    unsigned (*size) (struct _HashMap*);
    void (*first) (struct _HashMap*);
    Pair* (*next) (struct _HashMap*);
    void (*set_hash) (struct _HashMap*, HashMapHash);
    void (*set_compare) (struct _HashMap*, HashMapCompare);
    void (*set_clean_key) (struct _HashMap*, HashMapCleanKey);
    void (*set_clean_value) (struct _HashMap*, HashMapCleanValue);
} HashMap;

HashMap* HashMapInit();
void HashMapDeinit(HashMap* obj);
bool HashMapPut(HashMap* self, void* key, void* value);
void* HashMapGet(HashMap* self, void* key);
bool HashMapContain(HashMap* self, void* key);
bool HashMapRemove(HashMap* self, void* key);
unsigned HashMapSize(HashMap* self);
void HashMapFirst(HashMap* self);
Pair* HashMapNext(HashMap* self);
void HashMapSetHash(HashMap* self, HashMapHash func);
void HashMapSetCompare(HashMap* self, HashMapCompare func);
void HashMapSetCleanKey(HashMap* self, HashMapCleanKey func);
void HashMapSetCleanValue(HashMap* self, HashMapCleanValue func);

/*===========================================================================*
 *                        The container private data                         *
 *===========================================================================*/
static const unsigned magic_primes[] = {
    769, 1543, 3079, 6151, 12289, 24593, 49157, 98317, 196613, 393241, 786433,
    1572869, 3145739, 6291469, 12582917, 25165843, 50331653, 100663319,
    201326611, 402653189, 805306457, 1610612741,
};
static const int num_prime = sizeof(magic_primes) / sizeof(unsigned);
static const double load_factor = 0.75;


typedef struct _SlotNode {
    Pair pair_;
    struct _SlotNode* next_;
} SlotNode;

struct _HashMapData {
    int size_;
    int idx_prime_;
    unsigned num_slot_;
    unsigned curr_limit_;
    unsigned iter_slot_;
    SlotNode** arr_slot_;
    SlotNode* iter_node_;
    HashMapHash func_hash_;
    HashMapCompare func_cmp_;
    HashMapCleanKey func_clean_key_;
    HashMapCleanValue func_clean_val_;
};

unsigned _HashMapHash(void* key);
int _HashMapCompare(void* lhs, void* rhs);
void _HashMapReHash(HashMapData* data);

HashMap* HashMapInit()
{
    HashMap* obj = (HashMap*)malloc(sizeof(HashMap));
    if (unlikely(!obj))
        return NULL;

    HashMapData* data = (HashMapData*)malloc(sizeof(HashMapData));
    if (unlikely(!data)) {
        free(obj);
        return NULL;
    }

    SlotNode** arr_slot = (SlotNode**)malloc(sizeof(SlotNode*) * magic_primes[0]);
    if (unlikely(!arr_slot)) {
        free(data);
        free(obj);
        return NULL;
    }
    unsigned i;
    for (i = 0 ; i < magic_primes[0] ; ++i)
        arr_slot[i] = NULL;

    data->size_ = 0;
    data->idx_prime_ = 0;
    data->num_slot_ = magic_primes[0];
    data->curr_limit_ = (unsigned)((double)magic_primes[0] * load_factor);
    data->arr_slot_ = arr_slot;
    data->func_hash_ = _HashMapHash;
    data->func_cmp_ = _HashMapCompare;
    data->func_clean_key_ = NULL;
    data->func_clean_val_ = NULL;

    obj->data = data;
    obj->put = HashMapPut;
    obj->get = HashMapGet;
    obj->contain = HashMapContain;
    obj->remove = HashMapRemove;
    obj->size = HashMapSize;
    obj->first = HashMapFirst;
    obj->next = HashMapNext;
    obj->set_hash = HashMapSetHash;
    obj->set_compare = HashMapSetCompare;
    obj->set_clean_key = HashMapSetCleanKey;
    obj->set_clean_value = HashMapSetCleanValue;

    return obj;
}

void HashMapDeinit(HashMap* obj)
{
    if (unlikely(!obj))
        return;

    HashMapData* data = obj->data;
    SlotNode** arr_slot = data->arr_slot_;
    HashMapCleanKey func_clean_key = data->func_clean_key_;
    HashMapCleanValue func_clean_val = data->func_clean_val_;

    unsigned num_slot = data->num_slot_;
    unsigned i;
    for (i = 0 ; i < num_slot ; ++i) {
        SlotNode* pred;
        SlotNode* curr = arr_slot[i];
        while (curr) {
            pred = curr;
            curr = curr->next_;
            if (func_clean_key)
                func_clean_key(pred->pair_.key);
            if (func_clean_val)
                func_clean_val(pred->pair_.value);
            free(pred);
        }
    }

    free(arr_slot);
    free(data);
    free(obj);
    return;
}

bool HashMapPut(HashMap* self, void* key, void* value)
{
    /* Check the loading factor for rehashing. */
    HashMapData* data = self->data;
    if (data->size_ >= data->curr_limit_)
        _HashMapReHash(data);

    /* Calculate the slot index. */
    unsigned hash = data->func_hash_(key);
    hash = hash % data->num_slot_;

    /* Check if the pair conflicts with a certain one stored in the map. If yes,
       replace that one. */
    HashMapCompare func_cmp = data->func_cmp_;
    SlotNode** arr_slot = data->arr_slot_;
    SlotNode* curr = arr_slot[hash];
    while (curr) {
        if (func_cmp(key, curr->pair_.key) == 0) {
            if (data->func_clean_key_)
                data->func_clean_key_(curr->pair_.key);
            if (data->func_clean_val_)
                data->func_clean_val_(curr->pair_.value);
            curr->pair_.key = key;
            curr->pair_.value = value;
            return true;
        }
        curr = curr->next_;
    }

    /* Insert the new pair into the slot list. */
    SlotNode* node = (SlotNode*)malloc(sizeof(SlotNode));
    if (unlikely(!node))
        return false;

    node->pair_.key = key;
    node->pair_.value = value;
    if (!(arr_slot[hash])) {
        node->next_ = NULL;
        arr_slot[hash] = node;
    } else {
        node->next_ = arr_slot[hash];
        arr_slot[hash] = node;
    }
    ++(data->size_);

    return true;
}

void* HashMapGet(HashMap* self, void* key)
{
    HashMapData* data = self->data;

    /* Calculate the slot index. */
    unsigned hash = data->func_hash_(key);
    hash = hash % data->num_slot_;

    /* Search the slot list to check if there is a pair having the same key
       with the designated one. */
    HashMapCompare func_cmp = data->func_cmp_;
    SlotNode* curr = data->arr_slot_[hash];
    while (curr) {
        if (func_cmp(key, curr->pair_.key) == 0)
            return curr->pair_.value;
        curr = curr->next_;
    }

    return NULL;
}

bool HashMapContain(HashMap* self, void* key)
{
    HashMapData* data = self->data;

    /* Calculate the slot index. */
    unsigned hash = data->func_hash_(key);
    hash = hash % data->num_slot_;

    /* Search the slot list to check if there is a pair having the same key
       with the designated one. */
    HashMapCompare func_cmp = data->func_cmp_;
    SlotNode* curr = data->arr_slot_[hash];
    while (curr) {
        if (func_cmp(key, curr->pair_.key) == 0)
            return true;
        curr = curr->next_;
    }

    return false;
}

bool HashMapRemove(HashMap* self, void* key)
{
    HashMapData* data = self->data;

    /* Calculate the slot index. */
    unsigned hash = data->func_hash_(key);
    hash = hash % data->num_slot_;

    /* Search the slot list for the deletion target. */
    HashMapCompare func_cmp = data->func_cmp_;
    SlotNode* pred = NULL;
    SlotNode** arr_slot = data->arr_slot_;
    SlotNode* curr = arr_slot[hash];
    while (curr) {
        if (func_cmp(key, curr->pair_.key) == 0) {
            if (data->func_clean_key_)
                data->func_clean_key_(curr->pair_.key);
            if (data->func_clean_val_)
                data->func_clean_val_(curr->pair_.value);

            if (!pred)
                arr_slot[hash] = curr->next_;
            else
                pred->next_ = curr->next_;

            free(curr);
            --(data->size_);
            return true;
        }
        pred = curr;
        curr = curr->next_;
    }

    return false;
}

unsigned HashMapSize(HashMap* self)
{
    return self->data->size_;
}

void HashMapFirst(HashMap* self)
{
    HashMapData* data = self->data;
    data->iter_slot_ = 0;
    data->iter_node_ = data->arr_slot_[0];
    return;
}

Pair* HashMapNext(HashMap* self)
{
    HashMapData* data = self->data;

    SlotNode** arr_slot = data->arr_slot_;
    while (data->iter_slot_ < data->num_slot_) {
        if (data->iter_node_) {
            Pair* ptr_pair = &(data->iter_node_->pair_);
            data->iter_node_ = data->iter_node_->next_;
            return ptr_pair;
        }
        ++(data->iter_slot_);
        if (data->iter_slot_ == data->num_slot_)
            break;
        data->iter_node_ = arr_slot[data->iter_slot_];
    }
    return NULL;
}

void HashMapSetHash(HashMap* self, HashMapHash func)
{
    self->data->func_hash_ = func;
}

void HashMapSetCompare(HashMap* self, HashMapCompare func)
{
    self->data->func_cmp_ = func;
}

void HashMapSetCleanKey(HashMap* self, HashMapCleanKey func)
{
    self->data->func_clean_key_ = func;
}

void HashMapSetCleanValue(HashMap* self, HashMapCleanValue func)
{
    self->data->func_clean_val_ = func;
}


/*===========================================================================*
 *               Implementation for internal operations                      *
 *===========================================================================*/
unsigned _HashMapHash(void* key)
{
    return (unsigned)(intptr_t)key;
}

int _HashMapCompare(void* lhs, void* rhs)
{
    if ((intptr_t)lhs == (intptr_t)rhs)
        return 0;
    return ((intptr_t)lhs > (intptr_t)rhs)? 1 : (-1);
}

void _HashMapReHash(HashMapData* data)
{
    unsigned num_slot_new;

    /* Consume the next prime for slot array extension. */
    if (likely(data->idx_prime_ < (num_prime - 1))) {
        ++(data->idx_prime_);
        num_slot_new = magic_primes[data->idx_prime_];
    }
    /* If the prime list is completely consumed, we simply extend the slot array
       with treble capacity.*/
    else {
        data->idx_prime_ = num_prime;
        num_slot_new = data->num_slot_ * 3;
    }

    /* Try to allocate the new slot array. The rehashing should be canceled due
       to insufficient memory space.  */
    SlotNode** arr_slot_new = (SlotNode**)malloc(sizeof(SlotNode*) * num_slot_new);
    if (unlikely(!arr_slot_new)) {
        if (data->idx_prime_ < num_prime)
            --(data->idx_prime_);
        return;
    }

    unsigned i;
    for (i = 0 ; i < num_slot_new ; ++i)
        arr_slot_new[i] = NULL;

    HashMapHash func_hash = data->func_hash_;
    SlotNode** arr_slot = data->arr_slot_;
    unsigned num_slot = data->num_slot_;
    for (i = 0 ; i < num_slot ; ++i) {
        SlotNode* pred;
        SlotNode* curr = arr_slot[i];
        while (curr) {
            pred = curr;
            curr = curr->next_;

            /* Migrate each key value pair to the new slot. */
            unsigned hash = func_hash(pred->pair_.key);
            hash = hash % num_slot_new;
            if (!arr_slot_new[hash]) {
                pred->next_ = NULL;
                arr_slot_new[hash] = pred;
            } else {
                pred->next_ = arr_slot_new[hash];
                arr_slot_new[hash] = pred;
            }
        }
    }

    free(arr_slot);
    data->arr_slot_ = arr_slot_new;
    data->num_slot_ = num_slot_new;
    data->curr_limit_ = (unsigned)((double)num_slot_new * load_factor);
    return;
}

unsigned HashKey(void* key)
{
    return HashDjb2((char*)key);
}

int CompareKey(void* lhs, void* rhs)
{
    return strcmp((char*)lhs, (char*)rhs);
}

// void CleanKey(void* key)
// {
//     free(key);
// }

void CleanObject(void* obj)
{
    free(obj);
}

void CleanVector(void* vector)
{
    VectorDeinit((Vector*)vector);
}
///////////////////////////////////////////////////////////    End of HashMap

#include <pthread.h>
#include "mapreduce.h"

// Vector* vector;
HashMap* hashmap;
HashMap* key2elemindex;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

char* Get(char *key, int partition_number)
{
    Vector* vector = hashmap->get(hashmap, (void*)key);
    int elemindex = (int)(intptr_t)key2elemindex->get(key2elemindex, (void*)key);
    void* elem;
    // if (vector->next(vector, &elem))
    //     return (char*)elem;
    // else
    //     return NULL;
    if (elemindex < vector->size(vector)){
        vector->get(vector, elemindex, &elem);
        key2elemindex->put(key2elemindex, (void*)strdup(key), (void*)(intptr_t)++elemindex);// update
        return (char*)elem;
    } else {
        return NULL;
    }
}

typedef struct _MapperWrap {
    Mapper mapper;
    Vector* files;
} MapperWrap;

void* threadCallMultipleMapper(void* mapperwrap)
{
    Mapper mapper = ((MapperWrap*)mapperwrap)->mapper;
    Vector* files = ((MapperWrap*)mapperwrap)->files;
    void* elem;
    for (int i=0; i<files->size(files); i++){
        files->get(files, i, &elem);
        mapper((char*)elem);
    }
    return 0;
}

void MR_Run(int argc, char *argv[],\
	    Mapper mapper, int num_mappers,\
	    Reducer reduce, int num_reducers,\
	    Partitioner partition)
{
    // vector = VectorInit(32);
    hashmap = HashMapInit();// (char*, char*) pair
    HashMapSetHash(hashmap, HashKey);
    HashMapSetCompare(hashmap, CompareKey);
    HashMapSetCleanKey(hashmap, CleanObject);
    HashMapSetCleanValue(hashmap, CleanVector);

    key2elemindex = HashMapInit();// (char*, int) pair
    HashMapSetHash(key2elemindex, HashKey);
    HashMapSetCompare(key2elemindex, CompareKey);
    HashMapSetCleanKey(key2elemindex, CleanObject);
    // HashMapSetCleanValue(key2elemindex, CleanObject);

    Vector* mapperfiles = VectorInit(32);
    VectorSetClean(mapperfiles, CleanVector);
    Vector* files;
    for (int i=0; i<num_mappers; i++){
        files = VectorInit(32);
        mapperfiles->push_back(mapperfiles, files);
    }
    void* elem;
    for (int i=1; i<argc; i++){
        mapperfiles->get(mapperfiles, (i-1)%num_mappers, &elem);
        ((Vector*)elem)->push_back((Vector*)elem, (void*)argv[i]);
    }
   Vector* mapperwraps = VectorInit(32);
   VectorSetClean(mapperwraps, CleanObject);
    for (int i=0; i<num_mappers; i++){
        MapperWrap* mapperwrap = (MapperWrap*)malloc(sizeof(MapperWrap));
        mapperwrap->mapper = mapper;
        mapperfiles->get(mapperfiles, i, &elem);
        mapperwrap->files = (Vector*)elem;
        mapperwraps->push_back(mapperwraps, (void*)mapperwrap);
    }

    // mapper(argv[1]);
    pthread_t mapperthreads[num_mappers];
    for (int i=0; i<num_mappers; i++){
        mapperwraps->get(mapperwraps, i, &elem);
        pthread_create(&mapperthreads[i], NULL, threadCallMultipleMapper, (void*)elem);
    }
    for (int i=0; i<num_mappers; i++){
        pthread_join(mapperthreads[i], NULL);
    }

    printf ("hashmap size = %d", hashmap->size(hashmap));

    Vector* keys = VectorInit(32);// vector of (char*)
    void* key;
    hashmap->first(hashmap);
    Pair* ptr_pair;
    while ((ptr_pair = hashmap->next(hashmap)) != NULL){
        key = ptr_pair->key;
        Vector* vector = (Vector*)ptr_pair->value;

        keys->push_back(keys, (void*)key);
        key2elemindex->put(key2elemindex, (void*)strdup(key), (void*)(intptr_t)0);
        vector->sort(vector, CompareWord);
    }
    keys->sort(keys, CompareWord);

    // void* elem;
    for (int i=0; i<keys->size(keys); i++){
        keys->get(keys, i, &elem);
        reduce((char*)elem, Get, 0);
    }

    // VectorDeinit(vector);
    HashMapDeinit(hashmap);
    HashMapDeinit(key2elemindex);
    VectorDeinit(keys);
    VectorDeinit(mapperfiles);
    VectorDeinit(mapperwraps);
}

void MR_Emit(char *key, char *value)
{
    pthread_mutex_lock(&mutex);
    // vector->push_back(vector, (void*)value);
    Vector* vector;
    if (hashmap->contain(hashmap, (void*)key) == false){
        vector = VectorInit(32);
        // VectorSetClean(vector, CleanObject);
        hashmap->put(hashmap, (void*)strdup(key), (void*)vector);
    } else {
        vector = hashmap->get(hashmap, (void*)key);
    }
    // vector->push_back(vector, (void*)strdup(value));
    vector->push_back(vector, (void*)value);
    pthread_mutex_unlock(&mutex);
}

unsigned long MR_DefaultHashPartition(char *key, int num_partitions) {
    unsigned long hash = 5381;
    int c;
    while ((c = *key++) != '\0')
        hash = hash * 33 + c;
    return hash % num_partitions;
}