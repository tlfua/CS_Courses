
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.LinkedList;
//import java.util.List;
//import java.util.Map;
//import java.util.Queue;
//import java.util.Set;
//import java.util.Stack;
//import java.util.Iterator;
import java.util.Map.Entry;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;


/**
 * Filename:   CourseSchedulerUtil.java
 * Project:    p4
 * Authors:    Tien-Lung Fu
 * 
 * Use this class for implementing Course Planner
 * @param <T> represents type
 */

public class CourseSchedulerUtil<T> {
    
    // can add private but not public members
    
    /**
     * Graph object
     */
    private GraphImpl<T> graphImpl;
    private HashMap<T, ArrayList<T>> courseToPrerequisites;
    
    
    /**
     * constructor to initialize a graph object
     */
    public CourseSchedulerUtil() {
        this.graphImpl = new GraphImpl<T>();
    }
    
    /**
    * createEntity method is for parsing the input json file 
    * @return array of Entity object which stores information 
    * about a single course including its name and its prerequisites
    * @throws Exception like FileNotFound, JsonParseException
    */
    @SuppressWarnings("rawtypes")
    public Entity[] createEntity(String fileName) throws Exception {
    	Entity<String>[] etities = new Entity[100];
    	
    	JSONParser parser = new JSONParser();
        try {
            Object obj = parser.parse(new FileReader(fileName));
            JSONObject jsonObject = (JSONObject) obj;

            JSONArray courses = (JSONArray) jsonObject.get("courses");
            Iterator<JSONObject> c_iter = courses.iterator();
            JSONObject cur_c;
            int i = 0;
            while (c_iter.hasNext()) {
            	Entity<String> entity = new Entity<String>();
            	
            	cur_c = c_iter.next();
//            	System.out.println(cur_c.get("name"));
            	
            	entity.setName((String) cur_c.get("name"));
                
                JSONArray prerequisites_obj = (JSONArray) cur_c.get("prerequisites");
                Iterator<String> p_iter = prerequisites_obj.iterator();
                String[] prerequisites = new String[100];
                int j = 0;
                while (p_iter.hasNext()) {
//                	System.out.println(p_iter.next());
                	prerequisites[j] = p_iter.next();
                	j++;
                }
                entity.setPrerequisites(prerequisites);
                etities[i] = entity;
                i++;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    	
        return etities;
    }
    
    
    /**
     * Construct a directed graph from the created entity object 
     * @param entities which has information about a single course 
     * including its name and its prerequisites
     */
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public void constructGraph(Entity[] entities) {
    	
    	int i = 0;
    	while (entities[i] != null) {
//    		System.out.println(entities[i].getName());
    		if (!graphImpl.hasVertex((T) entities[i].getName()))
    			graphImpl.addVertex((T) entities[i].getName());
    		
    		T[] prerequisites = (T[]) entities[i].getPrerequisites();
    		int j = 0;
    		while (prerequisites[j] != null) {
    			
    			if (!graphImpl.hasVertex(prerequisites[j]))
    				graphImpl.addVertex((T) prerequisites[j]);
        		graphImpl.addEdge(prerequisites[j], (T) entities[i].getName());
    			j++;
    		}
    		i++;
    	}
    	graphImpl.printGraph();
//    	System.out.println("order=" + graphImpl.order() + ", size=" + graphImpl.size());
    	
    	
    	courseToPrerequisites = new HashMap<T, ArrayList<T>>();
    	i = 0;
    	while (entities[i] != null) {
    		if (!courseToPrerequisites.containsKey(entities[i].getName())) {
//        	    System.out.println(entities[i]);
    			courseToPrerequisites.put((T) entities[i].getName(), new ArrayList<T>());
    		}
    		
    		T[] prerequisites = (T[]) entities[i].getPrerequisites();
    		int j = 0;
    		while (prerequisites[j] != null) {
    			
    			if (!courseToPrerequisites.containsKey(prerequisites[j])) {
//            	    System.out.println(prerequisites[j]);
    				courseToPrerequisites.put((T) prerequisites[j], new ArrayList<T>());
    			}
    			ArrayList<T> arrayList = courseToPrerequisites.get((T) entities[i].getName());
        		arrayList.add(prerequisites[j]);
    			j++;
    		}
    		i++;
    	}
    }
 
    
    /**
     * Returns all the unique available courses
     * @return the sorted list of all available courses
     */
    public Set<T> getAllCourses() {
        return graphImpl.getAllVertices();
    }
 
    
    public ArrayList<T> topologicalSort() {
    	
    	HashMap<T, Integer> indegree = new HashMap();
    	Set<T> VerticesSet = graphImpl.getAllVertices();
    	
    	for (T v : VerticesSet)
    		indegree.put(v, 0);
    	
    	for (T v : VerticesSet) {
    		ArrayList<T> arrayList = (ArrayList<T>) graphImpl.getAdjacentVerticesOf(v);
    		for (T neibor : arrayList) {
//    			System.out.println(neibor);
    			indegree.put(neibor, indegree.get(neibor) + 1);
    		}		
    	}
    	
    	Queue<T> q = new LinkedList<T>();
    	for (T v : VerticesSet) {
    		if (indegree.get(v) == 0)
    			q.add(v);
    	}
    	
    	ArrayList<T> topologyOrder = new ArrayList<T>();
    	while (!q.isEmpty()) {
    		
    		T cur = q.poll();
    		topologyOrder.add(cur);
    		
    		ArrayList<T> neibors = (ArrayList<T>) graphImpl.getAdjacentVerticesOf(cur);
    		for (T neibor : neibors) {
    			indegree.put(neibor, indegree.get(neibor) - 1);
    			if (indegree.get(neibor) == 0)
    				q.add(neibor);
    		}
    	}
    	return topologyOrder;
    }
    
    
    /**
     * To check whether all given courses can be completed or not
     * @return boolean true if all given courses can be completed,
     * otherwise false
     * @throws Exception
     */
    public boolean canCoursesBeCompleted() throws Exception {
        if (graphImpl.order() == 0)
        	throw new Exception("The graph is not constructed yet!");        
        return getAllCourses().size() == topologicalSort().size();
    }
    
    
    /**
     * The order of courses in which the courses has to be taken
     * @return the list of courses in the order it has to be taken
     * @throws Exception when courses can't be completed in any order
     */
    public List<T> getSubjectOrder() throws Exception {
    	if (!canCoursesBeCompleted())
    		throw new Exception("Courses can't be completed in any order");        
        return topologicalSort();
    }

        
    /**
     * The minimum course required to be taken for a given course
     * @param courseName 
     * @return the number of minimum courses needed for a given course
     */
    public int getMinimalCourseCompletion(T courseName) throws Exception {
        
    	HashSet<T> seen = new HashSet<T>();
    	Queue<T> q = new LinkedList<T>();
    	q.add(courseName);
    	seen.add(courseName);
    	
    	int cnt = 0;
    	while (!q.isEmpty()) {
    		T cur = q.poll();
    		cnt++;
    		
    		ArrayList<T> arrayList = courseToPrerequisites.get(cur);
    		for (T node : arrayList) {
    			if (seen.contains(node))
    				throw new Exception("Cycle existed");
    			seen.add(node);
    			q.add(node);
    		}
    	}
    	return cnt - 1;
    }
    
}
