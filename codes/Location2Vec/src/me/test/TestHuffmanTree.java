package me.test;

import org.junit.Test;

import me.tree.HuffmanTree;
import me.tree.LocationData;

public class TestHuffmanTree {

	@Test
	public void testBuildTree() {
		HuffmanTree huffmanTree = new HuffmanTree();
		huffmanTree.buildTree("test.json");
		LocationData[] locations = huffmanTree.getAllLocations();
		for (LocationData location : locations) {
			System.out.println(location.toString());
		}
	}

}
