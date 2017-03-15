package me;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import me.tree.LocationData;
import me.util.JsonUtil;

/**
 * @author Duocai Wu
 * @date 2017年1月9日
 * @time 下午6:30:20
 *
 */
public class TrainFromTree {
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		JSONArray jsonArray = JsonUtil.getJArr("tree.json");
		LocationData[] los = new LocationData[jsonArray.length()];
		for (int i = 0; i < los.length; i++) {
			JSONObject ob;
			try {
				ob = jsonArray.getJSONObject(i);
				int[] points = getArray(ob,"points");
				int[] codes = getArray(ob, "codes");
				los[i] = new LocationData(ob.getInt("id"),
						points,codes);
				//test json
				System.out.println(los[i]);
			} catch (JSONException e) {
				e.printStackTrace();
			}
		}
 	}

	private static int[] getArray(JSONObject ob, String str) {
		JSONArray array = null;
		int[] points = null;
		try {
			array = ob.getJSONArray(str);
			points = new int[array.length()];
			for (int j = 0; j < points.length; j++) {
				points[j] = array.getInt(j);
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return points;
	}
	
}
