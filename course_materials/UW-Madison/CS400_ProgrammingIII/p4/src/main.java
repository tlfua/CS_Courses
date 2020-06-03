import java.util.*;
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.List;
//import java.util.Map;
//import java.util.Set;


public class main {
	
	public static void main(String[] args) throws Exception {

		CourseSchedulerUtil<String> courseSchedule = new CourseSchedulerUtil<String>();
	
		Entity[] entities = courseSchedule.createEntity("valid.json");
//		Entity[] entities = courseSchedule.createEntity("invalid.json");
//		
		courseSchedule.constructGraph(entities);
		
		Set<String> allCourses = courseSchedule.getAllCourses();
//		System.out.println(allCourses);
		
//		ArrayList<String> topologyOrder = courseSchedule.topologicalSort();
//		System.out.println(topologyOrder);
		
		try {
			System.out.println("Can be completed? " + courseSchedule.canCoursesBeCompleted());
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		try {
			ArrayList<String> res = (ArrayList<String>) courseSchedule.getSubjectOrder();
			System.out.println("subject order: " + res);
		} catch (Exception e) {
			e.printStackTrace();
		} 
		
		System.out.println("min # to before CS200 is " + courseSchedule.getMinimalCourseCompletion("CS200"));
		System.out.println("min # to before CS540 is " + courseSchedule.getMinimalCourseCompletion("CS540"));
		System.out.println("min # to before CS790 is " + courseSchedule.getMinimalCourseCompletion("CS790"));
		System.out.println("min # to before CS300 is " + courseSchedule.getMinimalCourseCompletion("CS300"));
		System.out.println("min # to before CS400 is " + courseSchedule.getMinimalCourseCompletion("CS400"));
		System.out.println("min # to before CS760 is " + courseSchedule.getMinimalCourseCompletion("CS760"));
		
//		try {
//			System.out.println("min # to before CS400 is " + courseSchedule.getMinimalCourseCompletion("CS400"));
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
//		
//		try {
//			System.out.println("min # to before CS300 is " + courseSchedule.getMinimalCourseCompletion("CS300"));
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
	}
}
