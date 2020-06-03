/**
 * Filename:   TestPQ.java
 * Project:    p1TestPQ
 * Authors:    Debra Deppeler, Tien-Lung Fu
 *
 * Semester:   Fall 2018
 * Course:     CS400
 * Lecture:    004 / Class Number: 46373
 *
 * Note: Warnings are suppressed on methods that construct new instances of
 * generic PriorityQueue types.  The exceptions that occur are caught and
 * such types are not able to be tested with this program.
 *
 * Due Date:   Before 10pm on September 18, 2018
 * Version:    2.0
 *
 * Credits:    for generating random strings (see generateUnicodeString below):
 * 			   https://www.baeldung.com/java-random-string
 *
 * Bugs:       no known bugs
 */

import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Random;

/**
 * Runs black-box unit tests on the priority queue implementations
 * passed in as command-line arguments (CLA).
 *
 * If a class with the specified class name does not exist
 * or does not implement the PriorityQueueADT interface,
 * the class name is reported as unable to test.
 *
 * If the class exists, but does not have a default constructor,
 * it will also be reported as unable to test.
 *
 * If the class exists and implements PriorityQueueADT,
 * and has a default constructor, the tests will be run.
 *
 * Successful tests will be reported as passed.
 *
 * Unsuccessful tests will include:
 *     input, expected output, and actual output
 *
 * Example Output:
 * Testing priority queue class: PQ01
 *    5 PASSED
 *    0 FAILED
 *    5 TOTAL TESTS RUN
 * Testing priority queue class: PQ02
 *    FAILED test00isEmpty: unexpectedly threw java.lang.NullPointerException
 *    FAILED test04insertRemoveMany: unexpectedly threw java.lang.ArrayIndexOutOfBoundsException
 *    3 PASSED
 *    2 FAILED
 *    5 TOTAL TESTS RUN
 *
 *   ... more test results here
 *
 * @author deppeler
 */
public class TestPQ {

	// set to true to see call stack trace for exceptions
	private static final boolean DEBUG = true;

	/**
	 * Run tests to determine if each Priority Queue implementation
	 * works as specified. User names the Priority Queue types to test.
	 * If there are no command-line arguments, nothing will be tested.
	 *
	 * @param args names of PriorityQueueADT implementation class types
	 * to be tested.
	 */
	public static void main(String[] args) {
		for (int i=0; i < args.length; i++)
			test(args[i]);

		if ( args.length < 1 )
			print("no PQs to test");
	}

	/**
	 * Run all tests on each priority queue type that is passed as a classname.
	 *
	 * If constructing the priority queue in the first test causes exceptions,
	 * then no other tests are run.
	 *
	 * @param className the name of the class that contains the
	 * priority queue implementation.
	 */
	private static void test(String className) {
		print("Testing priority queue class: "+className);
		int passCount = 0;
		int failCount = 0;
		try {

			if (test00isEmpty(className)) passCount++; else failCount++;
			if (test01getMaxEXCEPTION(className)) passCount++; else failCount++;

			if (test02removeMaxEXCEPTION(className)) passCount++; else failCount++;
			if (test03insertRemoveOne(className)) passCount++; else failCount++;
			if (test04insertRemoveMany(className)) passCount++; else failCount++;

			if (test05duplicatesAllowed(className)) passCount++; else failCount++;
			if (test06manyDataItems(className)) passCount++; else failCount++;

			// add calls to your additional test methods here
			if (test07getMaxEqremoveMax(className)) passCount++; else failCount++;
			if (test08removeAllisEmpty(className)) passCount++; else failCount++;
			if (test09duplicatesRemoveMaxTillEmpty(className)) passCount++; else failCount++;

			String passMsg = String.format("%4d PASSED", passCount);
			String failMsg = String.format("%4d FAILED", failCount);
			print(passMsg);
			print(failMsg);

		} catch (ClassNotFoundException | InstantiationException | IllegalAccessException e) {
			if (DEBUG) e.printStackTrace();
			print(className + " FAIL: Unable to construct instance of " + className);
		} finally {
			String msg = String.format("%4d TOTAL TESTS RUN", passCount+failCount);
			print(msg);
		}

	}

	/////////////////////////
	// ADD YOUR TEST METHODS HERE
	// Must test each operation of the PriorityQueueADT
	// Find and report cases that cause problems.
	// Do not try to fix or debug implementations.
	/////////////////////////

	/**
	 * Check if removeMax can clear the priority queue when duplicate values are inserted 10 times,
	 * and then removeMax is called 10 time too.
	 *
	 * @param className name of the priority queue implementation to test.
	 * @return true if removeMax works correctly with duplicate values.
	 * @throws InstantiationException
	 * @throws IllegalAccessException
	 * @throws ClassNotFoundException
	 */
	private static boolean test09duplicatesRemoveMaxTillEmpty(String className)
		throws InstantiationException, IllegalAccessException, ClassNotFoundException {
		PriorityQueueADT<Integer> pq = newIntegerPQ(className);
		try {
			// insert the same integer 10 times
			for (int i = 0; i < 10; ++i) {
				pq.insert(123);
			}
			// call removeMax 10 times
			for (int i = 0; i < 10; ++i) {
				pq.removeMax();
			}
			// check if pq is empty
			if (!pq.isEmpty()) {
				print("FAILED test09duplicatesRemoveMaxTillEmpty: removeMax on duplicate values does not work properly to clear the priority queue.");
				return false;
			}
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test09duplicatesRemoveMaxTillEmpty: unexpectedly threw " + e.getClass().getName());
			return false;
		}
		// no exception raised, test passed
		return true;
	}

	/**
	 * Confirm isEmpty is true after removing all elements
	 *
	 * @param className name of the priority queue implementation to test.
	 * @return true if isEmpty is true after removing all elements
	 * @throws InstantiationException
	 * @throws IllegalAccessException
	 * @throws ClassNotFoundException
	 */
	private static boolean test08removeAllisEmpty(String classname)
		throws InstantiationException, IllegalAccessException, ClassNotFoundException {

		PriorityQueueADT<Integer> pq = newIntegerPQ(classname);
		int x = 100;

		try {
			pq.insert(x);
			pq.removeMax();
		} catch (NoSuchElementException e) {
			return false;
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test08removeAllisEmpty: unexpectedly threw " + e.getClass().getName());
			return false;
		}

		try {
			if (pq.isEmpty()) {
				return true;
			} else {
				print ("FAILED test08removeAllisEmpty: PriorityQueue is not empty");
				return false;
			}
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test08removeAllisEmpty: unexpectedly threw " + e.getClass().getName());
			return false;
		}
	}

	/**
	 * Confirm that return the same value when calling getMax and removeMax.
	 *
	 * @param className name of the priority queue implementation to test.
	 * @return true if getMax and removeMax return the same value.
	 * @throws InstantiationException
	 * @throws IllegalAccessException
	 * @throws ClassNotFoundException
	 */
	private static boolean test07getMaxEqremoveMax(String classname)
		throws InstantiationException, IllegalAccessException, ClassNotFoundException {

		PriorityQueueADT<Integer> pq = newIntegerPQ(classname);
		int x = 100, y1, y2;

		try {
			pq.insert(x);
			y1 = pq.getMax();
			y2 = pq.removeMax();
		} catch (NoSuchElementException e) {
			return false;
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test07getMaxEqremoveMax: unexpectedly threw " + e.getClass().getName());
			return false;
		}

		if (y1 == y2) {
			return true;
		} else {
			print ("FAILED test07getMaxEqremoveMax: getMax val and removeMax val are different");
			return false;
		}
	}

	/**
	 * Confirm that the implementation can handle a sufficiently
	 * large number of items (1 million) without failing.
	 *
	 * @param className name of the priority queue implementation to test.
	 * @return true if the internal implementation array can expand and handle a sufficiently large number of items (1 million) without failing.
	 * @throws InstantiationException
	 * @throws IllegalAccessException
	 * @throws ClassNotFoundException
	 */
	private static boolean test06manyDataItems(String classname)
		throws InstantiationException, IllegalAccessException, ClassNotFoundException {

		// only test for integer in this test
		PriorityQueueADT<Integer> pq = newIntegerPQ(classname);
		try {
			// insert 1 million items into the priority queue
			for (int i = 0; i < 1000000; ++i) {
				pq.insert(i);
			}
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test06manyDataItems: unexpectedly threw " + e.getClass().getName());
			return false;
		}
		// no exception raised, test passed
		return true;
	}

	/**
	 * Inserts and removeMax duplicate items multiple times into priority queue,
	 * and check if duplicate values can be handled correctly.
	 *
	 * @param className name of the priority queue implementation to test.
	 * @return true if duplicate values can be inserted and removed correctly.
	 * @throws InstantiationException
	 * @throws IllegalAccessException
	 * @throws ClassNotFoundException
	 */
	private static boolean test05duplicatesAllowed(String classname)
		throws InstantiationException, IllegalAccessException, ClassNotFoundException {

		PriorityQueueADT<Integer> pq = newIntegerPQ(classname);
		int x = 100, y1, y2;

		try {
			pq.insert(x);
			pq.insert(x);
			y1 = pq.removeMax();
			y2 = pq.removeMax();
		} catch (NoSuchElementException e) {
			return false;
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test05duplicatesAllowed: unexpectedly threw " + e.getClass().getName());
			return false;
		}

		if (x == y1 && x == y2) {
			return true;
		} else {
			print ("FAILED test05duplicatesAllowed: duplicate values are not able to be inserted (and then removed)");
			return false;
		}
	}

	/**
	 * Inserts multiple elements and remove all of them to see if removed items are in descending order.
	 *
	 * @param className name of the priority queue implementation to test.
	 * @return true if removed items are in descending order.
	 * @throws ClassNotFoundException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static boolean test04insertRemoveMany(String className)
		throws InstantiationException, IllegalAccessException, ClassNotFoundException {

		PriorityQueueADT<Integer> pq = newIntegerPQ(className);
		int x1 = 100, x2 = 50;
		int y1, y2;

		try {
			pq.insert(x2);
			pq.insert(x1);
			y1 = pq.removeMax();
			y2 = pq.removeMax();
		} catch (NoSuchElementException e) {
			return false;
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test04insertRemoveMany: unexpectedly threw " + e.getClass().getName());
			return false;
		}

		if (y1 == x1 && y2 == x2) {
			return true;
		} else {
			print ("FAILED test04insertRemoveMany: removeMax does not return the max values in the priority order");
			return false;
		}
	}

	/**
	 * Insert one item and removeMax immediately on an empty priority queue
	 * to check if the removed item equals the inserted one.
	 * Repeat and test multiple test cases for Integer, String, Double priority queue.
	 *
	 * @param className name of the priority queue implementation to test.
	 * @return true if the element returned by removeMax equals to the inserted one for all test cases.
	 * @throws ClassNotFoundException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static boolean test03insertRemoveOne(String className)
		throws InstantiationException, IllegalAccessException, ClassNotFoundException {

		PriorityQueueADT<Integer> pq = newIntegerPQ(className);
		int x = 100, y;

		// pq.insert(x);
		try {
			pq.insert(x);
			y = pq.removeMax();
		} catch (NoSuchElementException e) {
			return false;
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test03insertRemoveOne: unexpectedly threw " + e.getClass().getName());
			return false;
		}

		// print (y);
		if (x != y) {
			print("FAILED test03insertRemoveOne: removeMax is not the same value as was inserted");
			return false;
		} else {
			// print ("same");
			return true;
		}
	}

	/**
	 * Confirm that removeMax throws NoSuchElementException if called on
	 * an empty priority queue.  Any other exception indicates a fail.
	 * @param className name of the priority queue implementation to test.
	 * @return true if removeMax on empty priority queue throws NoSuchElementException
	 * @throws ClassNotFoundException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static boolean test02removeMaxEXCEPTION(String className)
		throws InstantiationException, IllegalAccessException, ClassNotFoundException {

		PriorityQueueADT<Integer> pq = newIntegerPQ(className);
		try {
			pq.removeMax();
		} catch (NoSuchElementException e) {
			return true;
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test02removeMaxEXCEPTION: unexpectedly threw " + e.getClass().getName());
			return false;
		}
		print("FAILED test02removeMaxEXCEPTION: removeMax did not throw NoSuchElement exception on newly constructed PQ");
		return false;
	}

	/** DO NOT EDIT -- provided as an example
	 * Confirm that getMax throws NoSuchElementException if called on
	 * an empty priority queue.  Any other exception indicates a fail.
	 *
	 * @param className name of the priority queue implementation to test.
	 * @return true if getMax on empty priority queue throws NoSuchElementException
	 * @throws ClassNotFoundException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static boolean test01getMaxEXCEPTION(String className)
		throws InstantiationException, IllegalAccessException, ClassNotFoundException {
		PriorityQueueADT<Integer> pq = newIntegerPQ(className);

		// pq.insert(100);
		try {
			pq.getMax();
			// System.out.println(pq.getMax());
		} catch (NoSuchElementException e) {
			return true;
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test01getMaxEXCEPTION: unexpectedly threw " + e.getClass().getName());
			return false;
		}
		print("FAILED test01getMaxEXCEPTION: getMax did not throw NoSuchElement exception on newly constructed PQ");
		return false;
	}

	/** DO NOT EDIT THIS METHOD
	 * @return true if able to construct Integer priority queue and
	 * the instance isEmpty.
	 *
	 * @throws ClassNotFoundException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private static boolean test00isEmpty(String className)
		throws InstantiationException, IllegalAccessException, ClassNotFoundException {
		PriorityQueueADT<Integer> pq = newIntegerPQ(className);
		try {
		if (pq.isEmpty())
			return true;
		} catch (Exception e) {
			if (DEBUG) e.printStackTrace();
			print("FAILED test00isEmpty: unexpectedly threw " + e.getClass().getName());
			return false;
		}
		print("FAILED test00isEmpty: isEmpty returned false on newly constructed PQ");
		return false;
	}

	/** DO NOT EDIT THIS METHOD
	 * Constructs a max Priority Queue of Integer using the class that is name.
	 * @param className The specific Priority Queue to construct.
	 * @return a PriorityQueue
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 * @throws ClassNotFoundException
	 */
	@SuppressWarnings({ "unchecked" })
	public static final PriorityQueueADT<Integer> newIntegerPQ(String className) throws InstantiationException, IllegalAccessException, ClassNotFoundException {
		Class<?> pqClass = Class.forName(className);
		Object obj = pqClass.newInstance();
		if (obj instanceof PriorityQueueADT) {
			return (PriorityQueueADT<Integer>) obj;
		}
		return null;
	}

	/** DO NOT EDIT THIS METHOD
	 * Constructs a max Priority Queue of Double using the class that is named.
	 * @param className The specific Priority Queue to construct.
	 * @return a PriorityQueue
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 * @throws ClassNotFoundException
	 */
	@SuppressWarnings({ "unchecked" })
	public static final PriorityQueueADT<Double> newDoublePQ(final String className) throws InstantiationException, IllegalAccessException, ClassNotFoundException {
		Class<?> pqClass = Class.forName(className);
		Object obj = pqClass.newInstance();
		if (obj instanceof PriorityQueueADT) {
			return (PriorityQueueADT<Double>) obj;
		}
		return null;
	}

	/** DO NOT EDIT THIS METHOD
	 * Constructs a max Priority Queue of String using the class that is named.
	 * @param className The specific Priority Queue to construct.
	 * @return a PriorityQueue
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 * @throws ClassNotFoundException
	 */
	@SuppressWarnings({ "unchecked" })
	public static final PriorityQueueADT<String> newStringPQ(final String className) throws InstantiationException, IllegalAccessException, ClassNotFoundException {
		Class<?> pqClass = Class.forName(className);
		Object obj = pqClass.newInstance();
		if (obj instanceof PriorityQueueADT) {
			return (PriorityQueueADT<String>) obj;
		}
		return null;
	}


	/** DO NOT EDIT THIS METHOD
	 * Write the message to the standard output stream.
	 * Always adds a new line to ensure each message is on its own line.
	 * @param message Text string to be output to screen or other.
	 */
	private static void print(String message) {
		System.out.println(message);
	}

}
