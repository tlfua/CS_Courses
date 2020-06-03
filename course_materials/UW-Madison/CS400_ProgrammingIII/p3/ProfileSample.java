import java.util.*;

public class ProfileSample {
    public static void main(String args[]) {
        if (args.length < 1) {
            System.out.println("Expected one argument: <num_elements>");
            System.exit(1);
        }
        long numElements = Long.parseLong(args[0]);
        ArrayList<Long> elements = new ArrayList<Long>();

        for (long i = 0; i < numElements; i++) {
            elements.add(i);    
        }
        System.out.println(String.format("inserted %d elements into the hash set", numElements));
    }
}
