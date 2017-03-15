package me.tree;

/**
 * @author Duocai Wu
 * @date 2017年1月9日
 * @time 下午6:02:10
 *
 */
public class LocationData {
	private int id;
	private int[] points;
	private int[] codes;
	
	public LocationData(int id, int[] points, int[] codes){
		this.id = id;
		this.points = points;
		this.codes = codes;
	}

	public int[] getPoints() {
		return points;
	}

	public int getId() {
		return id;
	}
	
	@Override
	public String toString() {
		String str = "";
		str += "\nid: " + id;
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

	public int[] getCodes() {
		return codes;
	}
}
