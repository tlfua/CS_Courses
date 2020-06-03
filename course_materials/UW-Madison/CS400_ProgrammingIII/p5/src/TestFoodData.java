import static org.junit.Assert.*; 
import org.junit.Before;  // setUp method
import org.junit.After;   // tearDown method
import org.junit.Test;   

import java.util.*;


public class TestFoodData {

	// Allows us to create a new hash table before each test
	static FoodData fData;
		
	// TODO: add code that runs before each test method
	@Before
	public void setUp() throws Exception {
		fData = new FoodData();  
	}

	// TODO: add code that runs after each test method
	@After
	public void tearDown() throws Exception {
		fData = null;
	}
	
	
	@Test
	public void testcase_1_001() {
		fData.loadFoodItems("testcase_1.txt");
		ArrayList<String> in_rules = new ArrayList<>();
		
		in_rules.add("calories >= 2");
		assertEquals(fData.getNameStr(fData.filterByNutrients(in_rules)), "A\nB\nD\nE\n"); 
	}
	
	@Test
	public void testcase_1_002() {
		fData.loadFoodItems("testcase_1.txt");
		ArrayList<String> in_rules = new ArrayList<>();
		
		in_rules.add("calories <= 4");
		assertEquals(fData.getNameStr(fData.filterByNutrients(in_rules)), "B\nC\nD\nE\n"); 
	}
	
	@Test
	public void testcase_1_003() {
		fData.loadFoodItems("testcase_1.txt");
		ArrayList<String> in_rules = new ArrayList<>();
		
		in_rules.add("calories >= 2");
		in_rules.add("calories <= 4");
		assertEquals(fData.getNameStr(fData.filterByNutrients(in_rules)), "B\nD\nE\n"); 
	}
	
	// check muli foodItem wi same nutrient value
	@Test
	public void testcase_1_004() {
		fData.loadFoodItems("testcase_1.txt");
		ArrayList<String> in_rules = new ArrayList<>();
		
		in_rules.add("fat == 20");
		assertEquals(fData.getNameStr(fData.filterByNutrients(in_rules)), "A\nC\n");
	}
	
	@Test
	public void testcase_1_005() {
		fData.loadFoodItems("testcase_1.txt");
		ArrayList<String> in_rules = new ArrayList<>();
		
		in_rules.add("fat == 30");
		assertEquals(fData.getNameStr(fData.filterByNutrients(in_rules)), "B\nD\nE\n");
	}
	//////////////
	
	// check intersection between different rules
	@Test
	public void testcase_1_006() {
		fData.loadFoodItems("testcase_1.txt");
		ArrayList<String> in_rules = new ArrayList<>();
		
		in_rules.add("carbohydrate >= 2");
		in_rules.add("protein <= 4");
		in_rules.add("fiber <= 4");
		in_rules.add("protein >= 2");
		assertEquals(fData.getNameStr(fData.filterByNutrients(in_rules)), "B\nC\nD\n");
	}
	
	
	
}
