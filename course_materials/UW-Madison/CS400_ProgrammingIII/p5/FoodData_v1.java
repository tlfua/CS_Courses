//import java.util.HashMap;
//import java.util.List;
import java.util.*;
import java.util.regex.*;
import java.io.*; 

/**
 * This class represents the backend for managing all 
 * the operations associated with FoodItems
 * 
 * @author sapan (sapan@cs.wisc.edu)
 */
public class FoodData implements FoodDataADT<FoodItem> {
    
    // List of all the food items.
    private List<FoodItem> foodItemList;

    // Map of nutrients and their corresponding index
    private HashMap<String, BPTree<Double, List<FoodItem>>> indexes;
    
    private String [] basicNutrient = {"calories", "fat", "carbohydrate", "fiber", "protein"}; 
    
    
    /**
     * Public constructor
     */
    public FoodData() {
        // TODO : Complete
    	foodItemList = new ArrayList<>();
    	indexes = new HashMap<>();
    	
    	for (int i=0 ; i<5; i++)
    		indexes.put(basicNutrient[i], new BPTree<>(3));
    }
    
    
    /*
     * (non-Javadoc)
     * @see skeleton.FoodDataADT#loadFoodItems(java.lang.String)
     */
    @Override
    public void loadFoodItems(String filePath) {
        // TODO : Complete
    	File file = new File(filePath); 
    	  
    	BufferedReader br;
    	String st; 
		try {
			br = new BufferedReader(new FileReader(file));
			while ((st = br.readLine()) != null) { 
//				System.out.println(st);
				String[] parts = st.split(",");
				
				if (parts.length != 12)
					continue;
				
				FoodItem fItem = new FoodItem(parts[0], parts[1]);
				for (int i=2; i<=10; i=i+2) {
					fItem.addNutrient(parts[i], Double.parseDouble(parts[i+1]));
				}
			
				foodItemList.add(fItem);
				
				for (int i=0 ; i<5; i++) {
					BPTree<Double, List<FoodItem>> bpTree = indexes.get(basicNutrient[i]);
//					bpTree.insert(fItem.getNutrientValue(basicNutrient[i]), fItem);
					
					List<List<FoodItem>> valueLists = bpTree.rangeSearch(fItem.getNutrientValue(basicNutrient[i]), "==");
					List<FoodItem> valueList = null;
					if (valueLists.size() == 0) {
						valueList = new ArrayList<>();
						bpTree.insert(fItem.getNutrientValue(basicNutrient[i]), valueList);
					} else if (valueLists.size() == 1) {
						valueList = valueLists.get(0);
					}
					valueList.add(fItem);
				}
			}
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
    	
//		System.out.println("foodItemList len = " + foodItemList.size());
    	   
    }

    /*
     * (non-Javadoc)
     * @see skeleton.FoodDataADT#filterByName(java.lang.String)
     */
    @Override
    public List<FoodItem> filterByName(String substring) {
        // TODO : Complete
    	List<FoodItem> res = new ArrayList<>();
    	String p = "(?i:.*" + substring + ".*)";
//    	String p = substring;
    	
    	for (FoodItem fItem : foodItemList) {
//    		Pattern.matches(p, fItem.getName());
    		if (fItem.getName().matches(p))
    			res.add(fItem);
    	}
        return res;
    }

    /*
     * (non-Javadoc)
     * @see skeleton.FoodDataADT#filterByNutrients(java.util.List)
     */
    @Override
    public List<FoodItem> filterByNutrients(List<String> rules) {
        // TODO : Complete
    	HashMap<String, Double []> nutrient2range = new HashMap<>();
    	
    	for (String rule : rules) {
    		String [] parts = rule.split(" ");
    		Double [] bounds; // bounds[0]: equal, bounds[1]: lower bound, bounds[2]: upper bound
    		
    		
    		// parts[0]: nutrient name,  parts[1]: comparator,  parts[2]: value
    		if (!nutrient2range.containsKey(parts[0])) {
    			bounds = new Double [3];
    			bounds[0] = -1.0;
    			bounds[1] = -1.0;
    			bounds[2] = -1.0;
    			nutrient2range.put(parts[0], bounds);
    		}
    		else {
    			bounds = nutrient2range.get(parts[0]);
    		}
    		
    		if (bounds[0] != -1.0)
    			continue;
    		// 
    		else if (parts[1].charAt(0) == '=')
    			bounds[0] = Double.parseDouble(parts[2]);
    		else if (parts[1].charAt(0) == '>')
    			bounds[1] = Double.parseDouble(parts[2]);
    		else if (parts[1].charAt(0) == '<')
    			bounds[2] = Double.parseDouble(parts[2]);
    	}
    	
    	ArrayList<HashSet<FoodItem>> resSets = new ArrayList<>();
    	// traverse nutrient2range
    	for (Map.Entry<String, Double []> entry : nutrient2range.entrySet()) {
    	    String nutrient = entry.getKey();
    	    Double [] bounds = entry.getValue();
    	    
    	    if (bounds[0] != -1.0)
    	    	resSets.add(list2set(listlist2list(indexes.get(nutrient).rangeSearch(bounds[0], "=="))));
    	    else if (bounds[1] != -1.0 && bounds[2] == -1.0)
    	    	resSets.add(list2set(listlist2list(indexes.get(nutrient).rangeSearch(bounds[1], ">="))));
    	    else if (bounds[1] == -1.0 && bounds[2] != -1.0)
    	    	resSets.add(list2set(listlist2list(indexes.get(nutrient).rangeSearch(bounds[2], "<="))));
    	    else
    	    	resSets.add(list2set(listlist2list(indexes.get(nutrient).rangeSearch2(bounds[1], "INCLUSIVE", bounds[2], "INCLUSIVE"))));
    	}
    	
    	List<FoodItem> rt;
    	if (resSets.size() == 0)
//    		return new ArrayList<>();
    		rt = new ArrayList<>();
    	else if (resSets.size() == 1)
//    		return set2list(resSets.get(0));
    		rt = set2list(resSets.get(0));
    	else {
    		HashSet<FoodItem> tmp = resSets.get(0);
    		for (int i=1; i<resSets.size(); i++)
    			tmp.retainAll(resSets.get(i));
    		rt = set2list(tmp);
    	}
    	
//    	for (FoodItem fItem : rt)
//    		System.out.println(fItem.getName());
        return rt;
    }
    
    public List<FoodItem> listlist2list(List<List<FoodItem>> ll) {
    	List<FoodItem> res = new ArrayList<>();
    	for (List<FoodItem> l : ll)
    		for (FoodItem f : l)
    			res.add(f);
		return res;
    }
    
    public HashSet<FoodItem> list2set(List<FoodItem> list) {
    	HashSet<FoodItem> res = new HashSet<>();
    	for (FoodItem fItem : list)
    		res.add(fItem);
    	return res;
    }
	
    public List<FoodItem> set2list(HashSet<FoodItem> set) {
    	List<FoodItem> res = new ArrayList<>();
    	for (FoodItem fItem : set)
    		res.add(fItem);
    	
    	res.sort(Comparator.comparing(FoodItem::getName));
    	
		return res;	
    }
    
    public String getNameStr(List<FoodItem> list) {
    	String res = "";
    	for (FoodItem fItem : list)
//    		res += fItem.getName() + '_';
    		res += fItem.getName() + '\n';
    	return res;
    }

    /*
     * (non-Javadoc)
     * @see skeleton.FoodDataADT#addFoodItem(skeleton.FoodItem)
     */
    @Override
    public void addFoodItem(FoodItem foodItem) {
        // TODO : Complete
    }

    /*
     * (non-Javadoc)
     * @see skeleton.FoodDataADT#getAllFoodItems()
     */
    @Override
    public List<FoodItem> getAllFoodItems() {
        // TODO : Complete
        return null;
    }


	@Override
	public void saveFoodItems(String filename) {
		// TODO Auto-generated method stub
		
	}
	
	
	public static void main(String[] args) {
		
		FoodData fData = new FoodData();
		
		fData.loadFoodItems("foodItems.txt");
		
//		fData.loadFoodItems("testcase_1.txt");
		
//		ArrayList<String> in_rules = new ArrayList<>();
//		in_rules.add("protein == 0");
//		System.out.println(fData.getNameStr(fData.filterByNutrients(in_rules)));
		
		System.out.println(fData.getNameStr(fData.filterByName("co")));

		
	}

}
