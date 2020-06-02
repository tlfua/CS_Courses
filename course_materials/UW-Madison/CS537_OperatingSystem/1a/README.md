## **Implementation**

+ my-cat
  1. Iterately read each line of each file and store them in a array of string
  2. Print each element in the array line by line
+ my-sed
  1. Check argv[2] to decide whether to remove or replace a target word
  2. After seaching a line, if find no target word, store the original line
  3. If a target word appear in a line, explore the three cases - the target line the front, middle or end separately.
       Do relevant operation (romove or replace), and store 
  4. Print what are stored in the result array line by line
+ my-uniq
  1. While iterating the file, record current line an previous line
  2. If the current line is not equal to previous line, store the current line into result array
  3. Print what are stored in the result array line by line
