/**
 * Filename:   Profile.java
 * Project:    p3
 * Authors:    TODO: add your name(s) and lecture numbers here
 *
 * Semester:   Fall 2018
 * Course:     CS400
 * 
 * Due Date:   TODO: add assignment due date and time
 * Version:    1.0
 * 
 * Credits:    TODO: name individuals and sources outside of course staff
 * 
 * Bugs:       TODO: add any known bugs, or unsolved problems here
 */
import java.util.TreeMap;

public class Profile<K extends Comparable<K>, V> {

	HashTableADT<K, V> hashtable;
	TreeMap<K, V> treemap;
	
	// Instantiate hashtable and treemap
	public Profile() {
		hashtable = new HashTable<>();
		treemap = new TreeMap<>();
	}
	
	// Insert K, V into both hashtable and treemap
	public void insert(K key, V value) {
		hashtable.put(key, value);
		treemap.put(key, value);
	}
	
	// get value V for key K from both hashtable and treemap
	public void retrieve(K key) {
		hashtable.get(key);
		treemap.get(key);
	}
	
	public static void main(String[] args) {
		if (args.length < 1) {
			System.out.println("Expected 1 argument: <num_elements>");
			System.exit(1);
		}
		int numElements = Integer.parseInt(args[0]);
		
		/*
		 * TODO: complete the main method. 
		 * Create a profile object. 
		 * For example, Profile<Integer, Integer> profile = new Profile<Integer, Integer>();
		 * execute the insert method of profile as many times as numElements
		 * execute the retrieve method of profile as many times as numElements
		 * See, ProfileSample.java for example.
		 */
		Profile<Integer, Integer> profile = new Profile<Integer, Integer>();
		
		for (int i = 0; i < numElements; i++)
			profile.insert(i, i);
		
		for (int i = 0; i < numElements; i++)
			profile.retrieve(i);
		
		String msg = String.format("Successfully inserted and retreived %d elements into the hash table and treemap", numElements);
		System.out.println(msg);
	}
}
