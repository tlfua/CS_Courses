import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;


/**
 * Filename:   GraphImpl.java
 * Project:    p4
 * Course:     cs400 
 * Authors:    Tien-Lung Fu
 * Due Date:   
 * 
 * T is the label of a vertex, and List<T> is a list of
 * adjacent vertices for that vertex.
 *
 * Additional credits: 
 *
 * Bugs or other notes: 
 *
 * @param <T> type of a vertex
 */
public class GraphImpl<T> implements GraphADT<T> {

    // YOU MAY ADD ADDITIONAL private members
    // YOU MAY NOT ADD ADDITIONAL public members

    /**
     * Store the vertices and the vertice's adjacent vertices
     */
    private Map<T, List<T>> verticesMap; 
    
    
    /**
     * Construct and initialize and empty Graph
     */ 
    public GraphImpl() {
        verticesMap = new HashMap<T, List<T>>();
        // you may initialize additional data members here
    }

    public void addVertex(T vertex) {
    	verticesMap.put(vertex, new ArrayList<T>());
    }

    public void removeVertex(T vertex) {
    	for (Entry<T, List<T>> entry : verticesMap.entrySet()) {
    	    T key = entry.getKey();
    	    List<T> arrayList = entry.getValue();
    	    for (int i = 0; i < arrayList.size(); i++) { 		      
    	    	if (arrayList.get(i) == vertex)
    	    		arrayList.remove(i);
    	    }   		
    	}
    	verticesMap.remove(vertex);
    }

    public void addEdge(T vertex1, T vertex2) {        
    	ArrayList<T> arrayList = (ArrayList<T>) verticesMap.get(vertex1);
    	arrayList.add(vertex2);
    }
    
    public void removeEdge(T vertex1, T vertex2) {
    	ArrayList<T> arrayList = (ArrayList<T>) verticesMap.get(vertex1);
    	for (int i = 0; i < arrayList.size(); i++) { 		      
	    	if (arrayList.get(i) == vertex2)
	    		arrayList.remove(i);
	    }
    }    
    
    public Set<T> getAllVertices() {        
    	Set<T> allVertices = new HashSet<T>();    	
    	for (Entry<T, List<T>> entry : verticesMap.entrySet()) {
    	    T key = entry.getKey();
    	    allVertices.add(key);
    	    
    	    List<T> arrayList = entry.getValue();
    	    for (int i = 0; i < arrayList.size(); i++) { 		      
    	    	allVertices.add(arrayList.get(i));
    	    }   		
    	}
        return allVertices;
    }

    public List<T> getAdjacentVerticesOf(T vertex) {
        return verticesMap.get(vertex);
    }
    
    public boolean hasVertex(T vertex) {
    	return verticesMap.containsKey(vertex);
    }

    public int order() {        
        return verticesMap.size();
    }

    public int size() {
        int size = 0;
        for (Entry<T, List<T>> entry : verticesMap.entrySet()) {
    	    List<T> arrayList = entry.getValue();
    	    size += arrayList.size();
    	}
        return size;
    }
    
    
    /**
     * Prints the graph for the reference
     * DO NOT EDIT THIS FUNCTION
     * DO ENSURE THAT YOUR verticesMap is being used 
     * to represent the vertices and edges of this graph.
     */
    public void printGraph() {
//    	System.out.println("printGraph\n");
        for ( T vertex : verticesMap.keySet() ) {
            if ( verticesMap.get(vertex).size() != 0) {
                for (T edges : verticesMap.get(vertex)) {
                    System.out.println(vertex + " -> " + edges + " ");
                }
            } else {
                System.out.println(vertex + " -> " + " " );
            }
        }
    }
}
