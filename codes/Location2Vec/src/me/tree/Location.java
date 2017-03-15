package me.tree;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Duocai Wu
 * @date 2017年1月9日
 * @time 下午1:19:35
 *
 */
public class Location {
	private int id;
	private double latitude;
	private double longtitude;
	
	private Location parent;//父节点
	private List<Integer> codeReverse = new ArrayList<>();// huffman编码
	private List<Integer> pointReverse = new ArrayList<>();// huffman编码对应内节点的参数下标,从叶节点开始
	private int point;//当前节点参数位置
	private int code;//当前节点编码： 0 or 1

	/**
	 * 
	 * @param id - location id
	 * @param latitude
	 * @param longtitude
	 */
	public Location(int id, double latitude, double longtitude) {
		this.id = id;
		this.latitude = latitude;
		this.longtitude = longtitude;
	}
	
	/**
	 * Calculate the distance between lo1 and lo2
	 * @param lo1
	 * @param lo2
	 * @return - distance between lo1 and lo2
	 */
	public static double Distance(Location lo1, Location lo2) {
		// sqrt((x1-x2)^2 + (y1-y2)^2)
		return Math.sqrt(
					Math.pow(
							lo1.getLatitude()-lo2.getLatitude(),
							2)
					+
					Math.pow(
							lo1.getLongtitude() - lo2.getLongtitude(),
							2)
				);
	}
	
	/**
	 * add the parameter index i to the path of current location
	 * from leaf to  root
	 * @param i - parameter index
	 */
	public void addPoint(int i) {
		pointReverse.add(i);
	}
	
	public int[] getPoints() {
		int len = pointReverse.size();
		int[] points = new int[len];
		for (int i = 0; i < points.length; i++) {
			points[i] = pointReverse.get(len - 1 - i);
		}
		return points;
	}
	
	public int[] getCodes() {
		int len = codeReverse.size();
		int[] points = new int[len];
		for (int i = 0; i < points.length; i++) {
			points[i] = codeReverse.get(len - 1 - i);
		}
		return points;
	}
	
	/**
	 * add the code i to the path of current location
	 * from leaf to  root
	 * @param i - parameter index
	 */
	public void addCode(int i) {
		codeReverse.add(i);
	}
	
	/// get and set methods
	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public double getLatitude() {
		return latitude;
	}

	public void setLatitude(double latitude) {
		this.latitude = latitude;
	}

	public double getLongtitude() {
		return longtitude;
	}

	public void setLongtitude(double longtitude) {
		this.longtitude = longtitude;
	}

	public Location getParent() {
		return parent;
	}

	public void setParent(Location parent) {
		this.parent = parent;
	}
	
	public int getPoint() {
		return point;
	}

	public void setPoint(int point) {
		this.point = point;
	}
	
	@Override
	public String toString() {
		String str = "";
		str += "\nid: " + id;
		str += "\nlatitude: " + latitude;
		str += "\nlongitude: " + longtitude;
		str += "\npoints: ";
		int[] points = getPoints();
		for (int i = 0; i < points.length; i++) {
			str += points[i] + " ";
		}
		str += "\ncodes: ";
		int[] codes = getCodes();
		for (int i = 0; i < codes.length; i++) {
			str += codes[i] + " ";
		}
		return str;
	}

	public int getCode() {
		return code;
	}

	public void setCode(int code) {
		this.code = code;
	} 
}
