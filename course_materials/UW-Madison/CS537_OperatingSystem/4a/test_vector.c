#include "private.h"

int main()
{
    char words[5][10] = {"bbb", "ccc", "abc", "a", "d"};

    Vector* vector = VectorInit(32);
    VectorSetClean(vector, CleanObject);
    for (int i=0; i<5; i++){
        vector->push_back(vector, (void*)strdup(words[i]));
    }

    vector->sort(vector, CompareWord);    

    // void* elem;
    // for (int i=0; i<5; i++){
    //     vector->get(vector, i, &elem);
    //     printf("%s ", (char*)elem);
    // }

    // VectorDeinit(vector);

    char* names[3] = {"Alice\0", "Bob\0", "Chris\0"};

    /* We should initialize the container before any operations. */
    HashMap* map = HashMapInit();

    /* Set the custom hash value generator and key comparison functions. */
    HashMapSetHash(map, HashKey);
    HashMapSetCompare(map, CompareKey);

    /* If we plan to delegate the resource clean task to the container, set the
       custom clean functions. */
    HashMapSetCleanKey(map, CleanObject);
    HashMapSetCleanValue(map, CleanVector);

    char* key = strdup(names[0]);
    map->put(map, (void*)key, (void*)vector);

    map->first(map);
    Pair* ptr_pair;
    while ((ptr_pair = map->next(map)) != NULL){
        char* name = (char*)ptr_pair->key;
        Vector* value = (Vector*)ptr_pair->value;
        printf("%s\n", name);
        void* elem;
        for (int i=0; i<value->size(value); i++){
            vector->get(vector, i, &elem);
            printf("%s ", (char*)elem);
        }
    }

    HashMapDeinit(map);

    return 0;
}