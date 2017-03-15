package me;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import me.tree.HuffmanTree;
import me.tree.LocationData;
import me.util.JsonUtil;

/**
 * @author Duocai Wu
 * @date 2017年1月9日
 * @time 下午6:10:29
 *
 */
public class OutputLocationTree {
	private static String midLo = "mid_location.json";
	private static String smallLo = "small_location.json";
	private static final String originLo = "foursquare_location.json";

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		HuffmanTree huffmanTree = new HuffmanTree();
		huffmanTree.buildTree(originLo);
		
		HuffmanTree.println("getting data....");
		LocationData[] locations = huffmanTree.getAllLocations();
		HuffmanTree.println("outputting data....");
		try {
			BufferedWriter bw = new BufferedWriter(
					new FileWriter(new File("location_tree.v3.json")));
			//writeArray(bw,locations);
			writeObject(bw,locations);
			HuffmanTree.println("finish....");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void writeObject(BufferedWriter bw, LocationData[] locations) {
		try {
			bw.write("{\n");
			for (int i = 0; i < locations.length; i++) {
				LocationData locationData = locations[i];
				String out = "";
				
				//append id
				out += "\""+locationData.getId()+"\": {";
				
				//append points
				out += "\"points\": ";
				int[] points = locationData.getPoints();
				out += JsonUtil.getJsonStr(points) + ",";		
				
				//append codes
				out += "\"codes\": ";
				int[] codes = locationData.getCodes();
				out += JsonUtil.getJsonStr(codes);
				
				if (i == locations.length -1) {
					out += "}\n";
				}
				else {
					out += "},\n";
				}
				
				bw.write(out);
			}
			bw.write("}\n");
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void writeArray(BufferedWriter bw, LocationData[] locations) {
		try {
			bw.write("[\n");
			for (int i = 0; i < locations.length; i++) {
				LocationData locationData = locations[i];
				String out = "";
				out += "{";
				
				//append id
				out += "\"id\":"+"\""+locationData.getId()+"\",";
				
				//append points
				out += "\"points\": ";
				int[] points = locationData.getPoints();
				out += JsonUtil.getJsonStr(points) + ",";		
				
				//append codes
				out += "\"codes\": ";
				int[] codes = locationData.getCodes();
				out += JsonUtil.getJsonStr(codes);
				
				if (i == locations.length -1) {
					out += "}\n";
				}
				else {
					out += "},\n";
				}
				
				bw.write(out);
			}
			bw.write("]\n");
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}


