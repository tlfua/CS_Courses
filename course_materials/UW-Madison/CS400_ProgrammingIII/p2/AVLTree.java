/**
 * Filename:   AVLTree.java
 * Project:    p2
 * Authors:    Debra Deppeler, Tien-Lung Fu
 *
 * Semester:   Fall 2018
 * Course:     CS400
 * Lecture:    004 / Class Number: 46373
 *
 * Due Date:   specified in Canvas
 * Version:    1.0
 *
 * Credits:	   https://www.youtube.com/watch?v=gcULXE7ViZw
 *
 * Bugs:       no known bugs
 */

import java.lang.IllegalArgumentException;

/*
 * @param <K>: the abstract data type K
 */
public class AVLTree<K extends Comparable<K>> implements AVLTreeADT<K> {

	/** TODO: add class header comments here
	 * Represents a tree node.
	 * @param <K>
	 */
	class BSTNode<K> {
		/* fields */
		private K key;	// TODO: variable declaration comment
		private int height;	// TODO: variable declaration comment
		private BSTNode<K> left, right, parent;	// TODO: variable declaration comment

		/**
		 * Constructor for a BST node.
		 * @param key
		 */
		BSTNode(K key) {
			this.key = key;
			this.left = null;
			this.right = null;
			this.parent = null;
			this.height = 1;
		}
		/* accessors */
		public K getId() {
      		return key;
    	}
    	public BSTNode<K> getLeftChild() {
      		return left;
    	}
    	public BSTNode<K> getRightChild() {
      		return right;
    	}
    	public int getHeight() {
      		return height;
    	}
		/* mutators */
		public void setId(K key) {
      		this.key = key;
    	}
    	public void setLeftChild(BSTNode<K> node) {
      		this.left = node;
    	}
    	public void setRightChild(BSTNode<K> node) {
      		this.right = node;
    	}
    	public void setHeight(int height) {
      		this.height = height;
    	}
	}

	/*
	 * @param: None
	 * @return true if the BST is empty, otherwise return false
	 * @throws: None
	 */
	@Override
	public boolean isEmpty() {
		// TODO: implement isEmpty()
		if (root == null)
			return true;
		return false;
	}

	/*
	 * @param: the key to be inserted
	 * @return void
	 * @throws: DuplicateKeyException, IllegalArgumentException
	 */
	@Override
	public void insert(K key) throws DuplicateKeyException, IllegalArgumentException {
		// TODO: insert(K key)
		root = insertRecursive(root, null, key);
		updateHeight(updateHeightStart);
	}

	public BSTNode<K> insertRecursive(BSTNode<K> cur, BSTNode<K> parent, K key) {
		if (cur == null) {
			BSTNode<K> node = new BSTNode<K>(key);
			node.parent = parent;
			updateHeightStart = parent;
			return node;
		}

		if (cur.getId().compareTo(key) > 0) {
			cur.left = insertRecursive(cur.left, cur, key);
		}
		else if (cur.getId().compareTo(key) < 0) {
			cur.right = insertRecursive(cur.right, cur, key);
		}
		return cur;
	}

	public void updateHeight(BSTNode<K> cur) {

		int leftHeight, rightHeight;
		while (cur != null) {
			leftHeight = 0; rightHeight = 0;

			if (cur.left != null) {
				leftHeight = cur.left.height;
			}
			if (cur.right != null) {
				rightHeight = cur.right.height;
			}
			cur.height = Math.max(leftHeight, rightHeight) + 1;
			cur = cur.parent;
		}
	}

	/*
	 * @param: the key to be deleted
	 * @return void
	 * @throws: IllegalArgumentException
	 */
	@Override
	public void delete(K key) throws IllegalArgumentException {
		// TODO: delete(K key)
		root = deleteRecursive(root, null, key);
		updateHeight(updateHeightStart);
	}

	public BSTNode<K> findMin(BSTNode<K> cur) {
		prev = null;
		while (cur != null) {
			prev = cur;
			cur = cur.left;
		}
		return prev;
	}

	public BSTNode<K> deleteRecursive(BSTNode<K> cur, BSTNode<K> parent, K key) {

		if (cur == null)
			return root;
		else if (key.compareTo(cur.getId()) < 0)
			cur.left = deleteRecursive(cur.left, cur, key);
		else if (key.compareTo(cur.getId()) > 0)
			cur.right = deleteRecursive(cur.right, cur, key);
		else {
		// I found you, get ready to be deleted
			// Case 1: No Child
			if (cur.left == null && cur.right == null) {
				cur = null;
				updateHeightStart = parent;
			}
			// Case 2: One child
			else if (cur.left == null) {
				// BSTNode<K> tmp = cur;
				cur = cur.right;
				cur.parent = parent;
				// delete tmp;
				updateHeightStart = parent;
			}
			else if (cur.right == null) {
				cur = cur.left;
				cur.parent = parent;
				updateHeightStart = parent;
			}
			// Case 3: Two children
			else {
				BSTNode<K> tmp = findMin(cur.right);
				cur.setId(tmp.getId());
				cur.right = deleteRecursive(cur.right, cur, tmp.getId());
			}
		}
		return cur;
	}

	/*
	 * @param the key to be searched
	 * @return true if the key exist in the BST, otherwise return false
	 * @throws: IllegalArgumentException
	 */
	@Override
	public boolean search(K key) throws IllegalArgumentException {
		// TODO: search(K key)
		if (root == null)
			return false;
		return searchRecursive(root, key);
	}

	public boolean searchRecursive(BSTNode<K> cur, K key) {

		if (cur == null)
			return false;

		if (cur.getId().compareTo(key) == 0)
			return true;
		else if (cur.getId().compareTo(key) > 0)
            return searchRecursive(cur.left, key);
		else if (cur.getId().compareTo(key) < 0)
            return searchRecursive(cur.right, key);

		return false;
	}

	/*
	 * @param: None
	 * @return a inorder traversal string
	 * @throws: None
	 */
	@Override
	public String print() {
		// TODO: print()
		StringBuilder res = new StringBuilder();
		inorderPrint(root, res);
		return res.toString();
	}

	public void inorderPrint(BSTNode<K> cur, StringBuilder res) {
		if (cur == null)
			return;
		inorderPrint(cur.left, res);
		res.append("" + cur.getId());
		inorderPrint(cur.right, res);
	}

	/*
	 * @param: None
	 * @return true if the BST is balanced, otherwise return false
	 * @throws: None
	 */
	@Override
	public boolean checkForBalancedTree() {
		// TODO: checkForBalancedTree()
		BSTNode<K> cur = updateHeightStart;
		int leftHeight, rightHeight;
		while (cur != null) {
			leftHeight = 0; rightHeight = 0;

			if (cur.left != null) {
				leftHeight = cur.left.height;
			}
			if (cur.right != null) {
				rightHeight = cur.right.height;
			}

			if (Math.abs(leftHeight - rightHeight) > 1)
				return false;
			cur = cur.parent;
		}
		return true;
	}

	/*
	 * @param: None
	 * @return true if it is a valid BST, otherwise return false
	 * @throws: None
	 */
	@Override
	public boolean checkForBinarySearchTree() {
		// TODO: checkForBinarySearchTree()
		prev = null;
		return inorderCheckBST(root);
	}

	public boolean inorderCheckBST(BSTNode<K> cur) {
		if (cur == null)
			return true;

		if (!inorderCheckBST(cur.left))
			return false;
		if (prev != null && cur.getId().compareTo(prev.getId()) < 0)
			return false;
		prev = cur;
		return inorderCheckBST(cur.right);
	}

	// private data
	private BSTNode<K> root;
	private BSTNode<K> updateHeightStart;
	private BSTNode<K> prev;


	// draw
	public void printSideways() {
   		System.out.println("------------------------------------------");
   		recursivePrintSideways(root, "");
   		System.out.println("------------------------------------------");
 	}
 	public void recursivePrintSideways(BSTNode<K> current, String indent) {
   		if (current != null) {
     		recursivePrintSideways(current.getRightChild(), indent + "    ");
     		System.out.println(indent + current.getId());
     		recursivePrintSideways(current.getLeftChild(), indent + "    ");
   		}
 	}

 	public static void main(String[] arg) throws IllegalArgumentException, DuplicateKeyException {
        // first node
        AVLTree<Integer> tree = new AVLTree<Integer>();

        tree.insert(7);
        System.out.print(tree.checkForBalancedTree());
		System.out.print("\n");

        tree.insert(5);
        tree.insert(9);
        System.out.print(tree.checkForBalancedTree());
		System.out.print("\n");

        tree.insert(8);
        tree.insert(10);
        System.out.print(tree.checkForBalancedTree());
		System.out.print("\n");

        tree.insert(11);
        System.out.print(tree.checkForBalancedTree());
		System.out.print("\n");


//        tree.insert(6);
//        tree.insert(3);
//        tree.insert(2);
		tree.recursivePrintSideways(tree.root, "x");
		System.out.print("\n");

		System.out.print(tree.root.height);
		System.out.print("\n");


		System.out.print(tree.search(7));
		System.out.print(tree.search(3));
		System.out.print(tree.search(1));
		System.out.print(tree.search(8));
		System.out.print(tree.search(4));
		System.out.print(tree.search(-2));
		System.out.print("\n");

		System.out.print(tree.print());
		System.out.print("\n");

		System.out.print(tree.checkForBinarySearchTree());
		System.out.print("\n");

//		delete right leaf
		tree.delete(11);
		tree.recursivePrintSideways(tree.root, "x");
		System.out.print(tree.root.height);
		System.out.print("\n");
        System.out.print(tree.checkForBalancedTree());
		System.out.print("\n");
//
//		delete left leaf
//		tree.delete(2);
//		tree.recursivePrintSideways(tree.root, "x");

//		delete node w one left child
//		tree.delete(3);
//		tree.recursivePrintSideways(tree.root, "x");

//		delete node w one right child
//		tree.delete(10);
//		tree.recursivePrintSideways(tree.root, "x");

//		delete node w two children
		tree.delete(9);
		tree.recursivePrintSideways(tree.root, "x");
		System.out.print(tree.root.height);
		System.out.print("\n");
        System.out.print(tree.checkForBalancedTree());
		System.out.print("\n");


		tree.delete(5);
		tree.recursivePrintSideways(tree.root, "x");
		System.out.print(tree.root.height);
		System.out.print("\n");
        System.out.print(tree.checkForBalancedTree());
		System.out.print("\n");

		tree.delete(7);
		tree.recursivePrintSideways(tree.root, "x");
		System.out.print(tree.root.height);
		System.out.print("\n");
        System.out.print(tree.checkForBalancedTree());
		System.out.print("\n");
    }
}
