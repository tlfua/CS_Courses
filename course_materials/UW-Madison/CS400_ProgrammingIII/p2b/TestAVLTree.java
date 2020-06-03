/**
 * Filename:   TestAVLTree.java
 * Project:    p2
 * Authors:    Debra Deppeler, Tien-Lung Fu
 *
 * Semester:   Fall 2018
 * Course:     CS400
 * Lecture:    004 / Class Number: 46373
 * 
 * Due Date:   Before 10pm on September 24, 2018
 * Version:    1.0
 * 
 * Credits:    
 * 
 * Bugs:       no known bugs
 */

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.lang.IllegalArgumentException;
import org.junit.Test;

/** TODO: add class header comments here*/
public class TestAVLTree {

	/**
	 * Tests that an AVLTree is empty upon initialization.
	 */
	@Test
	public void test01isEmpty() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		assertTrue(tree.isEmpty());
		
		System.out.println("test01isEmpty pass");
	}

	/**
	 * Tests that an AVLTree is not empty after adding a node.
	 */
	@Test
	public void test02isNotEmpty() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(1);
			assertFalse(tree.isEmpty());
			
			System.out.println("test02isNotEmpty pass");
		} catch (DuplicateKeyException e) {
			System.out.println(e.getMessage());
		} catch (IllegalArgumentException e) {
			System.out.println(e.getMessage());
		}
	}
	
//	/**
//	 * Tests functionality of a single delete following several inserts.
//	 */
	@Test
	public void test03insertManyDeleteOne() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			for (int i = 1; i <= 100; i++) {
				tree.insert(i);
			}
			tree.delete(50);
			assertTrue(tree.checkForBalancedTree());
			
			System.out.println("test03insertManyDeleteOne pass");
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}
	
	/**
	 * Tests functionality of many deletes following several inserts.
	 */
	@Test
	public void test04insertManyDeleteMany() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			for (int i = 1; i <= 100; i++) {
				tree.insert(i);
			}
			for (int i = 2; i <= 100; i=i+2) {
				tree.delete(i);
			}
			assertTrue(tree.checkForBalancedTree());
			
			System.out.println("test04insertManyDeleteMany pass");
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}

	// TODO: add tests to completely test the functionality of your AVLTree class
	// Balance after interts
	@Test
	public void test05BalanceALeftLeftTreeAfterInterts() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(50);
			tree.insert(60);
			tree.insert(30);
			tree.insert(40);
			tree.insert(10);
			tree.insert(20);
//			tree.recursivePrintSideways(tree.root, "x");
			assertTrue(tree.checkForBalancedTree());
			
			System.out.println("test05BalanceALeftLeftTreeAfterInterts pass");
		} catch (DuplicateKeyException e) {
			System.out.println(e.getMessage());
		} catch (IllegalArgumentException e) {
			System.out.println(e.getMessage());
		}
	}
	
	@Test
	public void test06BalanceARightRightTreeAfterInterts() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(50);
			tree.insert(30);
			tree.insert(60);
			tree.insert(55);
			tree.insert(70);
			tree.insert(65);
//			tree.recursivePrintSideways(tree.root, "x");
			assertTrue(tree.checkForBalancedTree());
			
			System.out.println("test06BalanceARightRightTreeAfterInterts pass");
		} catch (DuplicateKeyException e) {
			System.out.println(e.getMessage());
		} catch (IllegalArgumentException e) {
			System.out.println(e.getMessage());
		}
	}
	
	@Test
	public void test07BalanceALeftRightTreeAfterInterts() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(50);
			tree.insert(60);
			tree.insert(30);
			tree.insert(40);
			tree.insert(10);
			tree.insert(35);
//			tree.recursivePrintSideways(tree.root, "x");
			assertTrue(tree.checkForBalancedTree());
			
			System.out.println("test07BalanceALeftRightTreeAfterInterts pass");
		} catch (DuplicateKeyException e) {
			System.out.println(e.getMessage());
		} catch (IllegalArgumentException e) {
			System.out.println(e.getMessage());
		}
	}
	
	@Test
	public void test08BalanceARightLeftTreeAfterInterts() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(50);
			tree.insert(30);
			tree.insert(60);
			tree.insert(55);
			tree.insert(70);
			tree.insert(57);
//			tree.recursivePrintSideways(tree.root, "x");
			assertTrue(tree.checkForBalancedTree());
			
			System.out.println("test08BalanceARightLeftTreeAfterInterts pass");
		} catch (DuplicateKeyException e) {
			System.out.println(e.getMessage());
		} catch (IllegalArgumentException e) {
			System.out.println(e.getMessage());
		}
	}
	// Balance after deletes
	@Test
	public void test09BalanceALeftLeftTreeAfterDeletes() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(50);
			tree.insert(60);
			tree.insert(30);
			tree.insert(40);
			tree.insert(10);
			tree.insert(70);
			tree.insert(20);
			tree.delete(70);
//			tree.recursivePrintSideways(tree.root, "x");
			assertTrue(tree.checkForBalancedTree());
			
			System.out.println("test09BalanceALeftLeftTreeAfterDeletes pass");
		} catch (DuplicateKeyException e) {
			System.out.println(e.getMessage());
		} catch (IllegalArgumentException e) {
			System.out.println(e.getMessage());
		}
	}

	@Test
	public void test10BalanceALeftRightTreeAfterDeletes() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(50);
			tree.insert(60);
			tree.insert(30);
			tree.insert(40);
			tree.insert(10);
			tree.insert(70);
			tree.insert(35);
			tree.delete(70);
//			tree.recursivePrintSideways(tree.root, "x");
			assertTrue(tree.checkForBalancedTree());
			
			System.out.println("test10BalanceALeftRightTreeAfterDeletes pass");
		} catch (DuplicateKeyException e) {
			System.out.println(e.getMessage());
		} catch (IllegalArgumentException e) {
			System.out.println(e.getMessage());
		}
	}
	
	@Test
	public void test11BalanceARightRightTreeAfterDeletes() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(50);
			tree.insert(30);
			tree.insert(60);
			tree.insert(55);
			tree.insert(70);
			tree.insert(20);
			tree.insert(65);
			tree.delete(20);
//			tree.recursivePrintSideways(tree.root, "x");
			assertTrue(tree.checkForBalancedTree());
			
			System.out.println("test11BalanceARightRightTreeAfterDeletes pass");
		} catch (DuplicateKeyException e) {
			System.out.println(e.getMessage());
		} catch (IllegalArgumentException e) {
			System.out.println(e.getMessage());
		}
	}

	@Test
	public void test12BalanceARightLeftTreeAfterDeletes() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(50);
			tree.insert(30);
			tree.insert(60);
			tree.insert(55);
			tree.insert(70);
			tree.insert(20);
			tree.insert(57);
			tree.delete(20);
//			tree.recursivePrintSideways(tree.root, "x");
			assertTrue(tree.checkForBalancedTree());
			
			System.out.println("test12BalanceARightLeftTreeAfterDeletes pass");
		} catch (DuplicateKeyException e) {
			System.out.println(e.getMessage());
		} catch (IllegalArgumentException e) {
			System.out.println(e.getMessage());
		}
	}
	// Null Exception
	@Test
	public void test13InsertNullException() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(null);
		} catch (IllegalArgumentException e) {
//			System.out.println("test13: IllegalArgumentException");
			System.out.println("test13InsertNullException pass");
		} catch (Exception e) {
			System.out.println("test13: other exceptions");
		}
	}
	
	@Test
	public void test14DeleteNullException() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.delete(null);
		} catch (IllegalArgumentException e) {
//			System.out.println("test14: IllegalArgumentException");
			System.out.println("test14DeleteNullException pass");
		} catch (Exception e) {
			System.out.println("test14: other exceptions");
		}
	}
	
	@Test
	public void test15SearchNullException() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.search(null);
		} catch (IllegalArgumentException e) {
//			System.out.println("test15: IllegalArgumentException");
			System.out.println("test15SearchNullException pass");
		} catch (Exception e) {
			System.out.println("test15: other exceptions");
		}
	}
	// DuplicateKeyException
	@Test
	public void test16InsertDuplicateKeyException() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(1);
			tree.insert(1);
		} catch (DuplicateKeyException e) {
//			System.out.println("test16: DuplicateKeyException");
			System.out.println("test16InsertDuplicateKeyException pass");
		} catch (Exception e) {
			System.out.println("test16: other exceptions");
		}
	}
	// Print
	@Test
	public void test17Print() {
		AVLTree<Integer> tree = new AVLTree<Integer>();
		try {
			tree.insert(5);
			tree.insert(4);
			tree.insert(3);
			tree.insert(2);
			tree.insert(1);
//			System.out.println(tree.print());
			assertEquals(tree.print(), "1 2 3 4 5 ");
			
			System.out.println("test17Print pass");
		} catch (Exception e) {
			System.out.println("test17: other exceptions");
		}
	}

}
