
import java.util.Queue;

public class RadixSort {


    public static int getDigitAtPlace(int number, int placeValue) {
        return number / placeValue % 10;
    }

    public static List<Integer> onePass(List<Integer> theList, int placeValue) {

        ArrayList<Queue> QList = new ArrayList<Queue>(10);

        for (int i = 0; i < theList.length(); i++) {
            q = QList[getDigitAtPlace(theList[i], placeValue)]
            q.
        }
    }


}