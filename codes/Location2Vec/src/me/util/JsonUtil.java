package me.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * @author Duocai Wu
 * @date 2017年1月9日
 * @time 下午2:18:54
 *
 */
public class JsonUtil {
	//test
	public static void main(String[] args) {
		JSONObject jsonObject = getJOb("test.json");
		@SuppressWarnings("unchecked")
		Iterator<String> keys =  jsonObject.keys();
		while (keys.hasNext()) {
			try {
				JSONObject jObject = jsonObject.getJSONObject(keys.next());
				System.out.println(jObject);
			} catch (JSONException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * get the json object from the file
	 * @param path - json file path 
	 * @return
	 */
	public static JSONObject getJOb(String path){
		File file = new File(path);
		BufferedReader bReader = null;
		String aString = "";
		try {
			bReader = new BufferedReader(
					new FileReader(file));
			while (bReader.ready()) {
				aString += bReader.readLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		JSONObject jsonObject = null;
		try {
			jsonObject = new JSONObject(aString);
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return jsonObject;	
	}
	
	/**
	 * get json array from file
	 * @param path - path of the json file
	 * @return
	 */
	public static JSONArray getJArr(String path){
		File file = new File(path);
		BufferedReader bReader = null;
		String aString = "";
		try {
			bReader = new BufferedReader(
					new FileReader(file));
			while (bReader.ready()) {
				aString += bReader.readLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		JSONArray jsonObject = null;
		try {
			jsonObject = new JSONArray(aString);
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return jsonObject;	
	}
	
	/**
	 * change the array to json string
	 * @param array
	 * @return
	 */
	public static String getJsonStr(int[] array) {
		String pointsStr = "[";
		for (int j = 0; j < array.length; j++) {
			pointsStr += "\""+array[j]+"\",";
		}
		//去掉最后一个逗号
		return pointsStr.substring(0, pointsStr.length()-1)+"]";
	}
}
