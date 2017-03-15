package me;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import me.tree.HuffmanTree;
import me.tree.Location;
import me.util.JsonUtil;

/**
 * @author Duocai Wu
 * @date 2017年1月9日
 * @time 下午9:26:46
 *
 */
public class OutputUser {
	private static final String smallUsers = "small_user.json";
	private static final String originUsers = "foursquare_user.json";

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println("loading users.....");
		JSONObject userOb = JsonUtil.getJOb(originUsers);
		@SuppressWarnings("unchecked")
		Iterator<String> keys =  userOb.keys();
		System.out.println("printing users.....");
		try {
			BufferedWriter bw = new BufferedWriter(
					new FileWriter(new File("user_new.json")));
			bw.write("[\n");
			while (keys.hasNext()) {
				String key = keys.next();
				JSONObject user = userOb.getJSONObject(key);
				String out = "{\"id\": \"" + key+"\",";
				JSONArray history = user.getJSONArray("location");
				String array = "\"locations\": [";
				for (int i = 0; i < history.length(); i++) {
					array += "\"" + history.getJSONObject(i)
							.getString("l_id") + "\",";
				}
				out += array.substring(0, array.length()-1) + "]}";
				if (keys.hasNext()) {
					out += ",";
				}
				bw.write(out+"\n");
			}
			bw.write("]\n");
			bw.close();
			System.out.println("finish.....");
		} catch (JSONException | IOException e) {
			e.printStackTrace();
		}
	}
}
