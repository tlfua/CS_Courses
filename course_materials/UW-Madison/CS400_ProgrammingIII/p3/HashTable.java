/**
 * Filename:   HashTable.java
 * Project:    p3
 * Authors:    Tien-Lung Fu
 *
 * Semester:   Fall 2018
 * Course:     CS400
 * 
 * Due Date:   
 * Version:    1.0
 * 
 * Credits:    
 * 
 * Bugs:       No known bugs
 */


import java.util.NoSuchElementException;
import java.util.ArrayList;

// The collision resolution: bucket
// The data structure for each bucket is ArrayList
// Hashing Algorithm: 
// use Java hashCode to get the bucketIndex.
// When performing put, 
// insert (key, value) pair into the front of the ArrayList with the corresponding bucketIndex.
// Similar operations for performing get and remove
// If the amount of the total elements (size) is larger than the shreshold,
// we increase the numBuckets to (2 * numBuckets + 1)
class HashNode<K, V> { 
    K key; 
    V value; 
  
    // Reference to next node 
    HashNode<K, V> next; 
  
    // Constructor 
    public HashNode(K key, V value) 
    { 
        this.key = key; 
        this.value = value; 
    } 
} 


public class HashTable<K extends Comparable<K>, V> implements HashTableADT<K, V> {
	
	/* 
	 * bucketArray: array of chains
	 * numBuckets: number of buckets
	 * size: size of (key, value) pairs
	 */
	private ArrayList<HashNode<K, V>> bucketArray;  
    private int numBuckets; 
    private int size;
    private double shreshold;
		
	// Constructor
	public HashTable() {
		bucketArray = new ArrayList<>();
		numBuckets = 31;
		size = 0;
		shreshold = 0.7;
		
		// for each bucket, create empty chain
		for (int i = 0; i < numBuckets; i++) {
			bucketArray.add(null);
		}
	}
	
	/* 
	 * Constructor
	 * @param: initialCapacity, shreshold: if loadFactor exceed this value, increase numBuckets
	*/
	public HashTable(int initialCapacity, double _shreshold) {
		bucketArray = new ArrayList<>();
		numBuckets = initialCapacity;
		size = 0;
		shreshold = _shreshold;
		
		// for each bucket, create empty chain
		for (int i = 0; i < numBuckets; i++) {
			bucketArray.add(null);
		}
	}

	/*
	 * @param: key, value
	 * @return: None
	 * @throws: IllegalArgumentException
	 */
	@Override
	public void put(K key, V value) throws IllegalArgumentException {
		if ((key == null) || (value == null)) {
			throw new IllegalArgumentException("Illegal key or value Exception");
		}
		
		int bucketIndex = getBucketIndex(key);
		HashNode<K, V> head = bucketArray.get(bucketIndex);
		
		// check if key is already present
		while (head != null) {
			if (head.key.equals(key)) {
				head.value = value;
				return;
			}
			head = head.next;
		}
		
		// Insert a new (key, value) pair
		size++;
		head = bucketArray.get(bucketIndex);
		HashNode<K, V> newNode = new HashNode<K, V>(key, value);
		newNode.next = head;
		bucketArray.set(bucketIndex, newNode);
		
		// if load factor goes beyond threshold, increase numBuckets
		if ((1.0 * size)/ numBuckets >= shreshold) {
			ArrayList<HashNode<K, V>> temp = bucketArray; 
            bucketArray = new ArrayList<>(); 
            numBuckets = 2 * numBuckets + 1; 
            size = 0; 
            for (int i = 0; i < numBuckets; i++) 
                bucketArray.add(null); 
  
            for (HashNode<K, V> headNode : temp) 
            { 
                while (headNode != null) 
                { 
                    put(headNode.key, headNode.value); 
                    headNode = headNode.next; 
                } 
            } 
		}
	}
	
	/*
	 * @param: key
	 * @return: value
	 * @throws: IllegalArgumentException, NoSuchElementException
	 */
	@Override
	public V get(K key) throws IllegalArgumentException, NoSuchElementException {
		if (key == null) {
			throw new IllegalArgumentException("Illegal key Exception");
		}
		
		int bucketIndex = getBucketIndex(key);
		HashNode<K, V> head = bucketArray.get(bucketIndex);
		
		while (head != null) {
			if (head.key.equals(key))
				return head.value;
			head = head.next;
		}
		
		throw new NoSuchElementException("Searched element does not exiset");
	}
	
	/*
	 * @param: key
	 * @return: None
	 * @throws: IllegalArgumentException, NoSuchElementException
	 */
	@Override
	public void remove(K key) throws IllegalArgumentException, NoSuchElementException {
		if (key == null) {
			throw new IllegalArgumentException("Illegal key Exception");
		}
		
		int bucketIndex = getBucketIndex(key);
		HashNode<K, V> head = bucketArray.get(bucketIndex);
		
		HashNode<K, V> prev = null;
		while (head != null) {
			if (head.key.equals(key))
				break;
			prev = head;
			head = head.next;
		}
		
		if (head == null)
			throw new NoSuchElementException("Deleted element does not exiset");
		
		size--;
		// re-connect, meaning deleting the memory pointed by head
		if (prev != null)
			prev.next = head.next;
		else
			bucketArray.set(bucketIndex, head.next);
	}
	
	// get the amount of total elements
	@Override
	public int size() {
		return size;
	}
		
	
	private int getBucketIndex(K key) {
		int hashCode = key.hashCode();
		int index = hashCode % numBuckets;
		return index;
	}
	
	public static void main(String[] args) { 
        HashTable<String, Integer> map = new HashTable<>();
        map.put("this", 1); 
        map.put("coder", 2); 
        map.put("hi", 3); 
//        System.out.println(map.get("this"));
        
        System.out.println(map.get("this"));
        System.out.println(map.size());
        map.remove("this");
        System.out.println(map.size());
    }
}
