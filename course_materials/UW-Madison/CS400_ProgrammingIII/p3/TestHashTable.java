/**
 * Filename:   TestHashTable.java
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

import java.util.NoSuchElementException; // expect to need
import static org.junit.Assert.*; 
import org.junit.Before;  // setUp method
import org.junit.After;   // tearDown method
import org.junit.Test;   

/** TODO: add class header comments here*/
public class TestHashTable{

	// TODO: add other fields that will be used by multiple tests
	
	// Allows us to create a new hash table before each test
	static HashTableADT<Integer, Object> ht;
	
	// TODO: add code that runs before each test method
	@Before
	public void setUp() throws Exception {
		ht = new HashTable<Integer, Object>();  
	}

	// TODO: add code that runs after each test method
	@After
	public void tearDown() throws Exception {
		ht = null;
	}
		
	/** IMPLEMENTED AS EXAMPLE FOR YOU
	 * Tests that a HashTable is empty upon initialization
	 */
	@Test
	public void test000_IsEmpty() {
		assertEquals("size with 0 entries:", 0, ht.size());
	}
	
	/** IMPLEMENTED AS EXAMPLE FOR YOU
	 * Tests that a HashTable is not empty after adding one (K, V) pair
	 */
	@Test
	public void test001_IsNotEmpty() {
		ht.put(1,"0001");
		int expected = 1;
		int actual = ht.size();
		assertEquals("size with one entry:",expected,actual);
	}

	/** IMPLEMENTED AS EXAMPLE FOR YOU
	 * Other tests assume <int,Object> pairs,
	 * this test checks that <Long,Object> pair also works.
	 */
	@Test
	public void test010_Long_Object() {
		Long key = 9876543210L;
		Object expected = "" + key;		
		HashTableADT<Long,Object> table = 
				new HashTable<Long,Object>();
		table.put(key, expected);
		Object actual = table.get(key);
		assertTrue("put-get of (Long,Object) pair",
				expected.equals(actual));
	}
	
	/*
	 * Tests that the value for a key is updated 
	 * when tried to insert again.
	 */
	@Test
	public void test011_Update() {
		ht = new HashTable<Integer, Object>();
		ht.put(1, "s1");
		ht.put(1, "s2");
		assertEquals("s2", ht.get(1));
	}
	
	/*
	 * Tests that inserting many and removing one entry
	 * from the hash table works
	 */
	@Test(timeout=1000 * 10)
	public void test100_InsertManyRemoveOne() {
		ht = new HashTable<Integer, Object>();
		for  (int i = 1; i < 50; i=i+2)
			ht.put(i, i+1);
		ht.remove(49);
		assertEquals(24, ht.size());
	}
	
	/*
	 * Tests ability to insert many entries and 
	 * and remove many entries from the hash table
	 */
	@Test(timeout=1000 * 10)
	public void test110_InsertRemoveMany() {
		ht = new HashTable<Integer, Object>();
		for  (int i = 1; i < 50; i=i+2)
			ht.put(i, i+1);
		for  (int i = 3; i < 50; i=i+2)
			ht.remove(i);
		assertEquals(1, ht.size());
	}
}
