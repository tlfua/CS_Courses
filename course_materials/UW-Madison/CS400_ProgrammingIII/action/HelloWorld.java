import java.util.*;
import java.util.Random;
import java.lang.*;
import java.io.*;

public class HelloWorld{
    public static void main(String[] args) {
        List<Integer> theList = new ArrayList(Arrays.asList(436,820,875,105,34,902,572,805,336,687,553,486,219,16,372));
        System.out.println(getDigitAtPlace(23456, 10));
        List<Integer> pass1 = onePass(theList, 1);
        List<Integer> pass2 = onePass(pass1, 10);
        System.out.println(onePass(pass2, 100));
    }

    public static int getDigitAtPlace(int number, int placeValue) {
        int digitNumber = (int) (number / placeValue);
        digitNumber = digitNumber % 10;
        return digitNumber;
    }

    public static List<Integer> onePass(List<Integer> theList, int placeValue) {
        List<Queue<Integer>> queueList = new ArrayList<Queue<Integer>>();
        List<Integer> returnList = new ArrayList<Integer>();
        for (int i = 0; i <= 9; i++) {
            queueList.add(new LinkedList<Integer>());
        }

        for (int i = 0; i < theList.size(); i++) {
            int index = getDigitAtPlace(theList.get(i), placeValue);
            Queue<Integer> q = queueList.get(index);
            q.offer(theList.get(i));
            // queueList.set(index, q);
        }

        for (int i = 0; i < queueList.size(); i++) {
            Queue<Integer> q = queueList.get(i);
            int size = q.size();
            for (int j = 0; j < size; j++) {
                returnList.add(q.poll());
            }
        }

        return returnList;
    }
}
