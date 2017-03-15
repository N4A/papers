package me.tree;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.json.JSONException;
import org.json.JSONObject;

import me.util.JsonUtil;
import me.util.SortUtil;

/**
 * @author Duocai Wu
 * @date 2017年1月9日
 * @time 下午1:18:50
 *
 */
public class HuffmanTree {
	private final String LAT = "latitude";
	private final String LON = "longitude";
	
	private int locNum; //地点数目
	private List<Location> allLos = new ArrayList<>();//原始所有地点
	
	private List<Location> curLos = new ArrayList<>();//构建树的过程中所有地点
	private List<Distance> distances = new CopyOnWriteArrayList<>();//地点之间的距离
	private int point = 0;
	
	private static long startTime;
	
	public HuffmanTree() {
		startTime = System.currentTimeMillis()/1000;
	}
	
	
	/**
	 * build huffman tree according to the location in the file
	 * 树的左枝以0标记，右枝以1标记
	 * @param filePath - location file path
	 */
	public void buildTree(String filePath) {
		println("init source.....");
		initLocations(filePath);
		println("size: "+locNum);
		println("calculate distance.....");
		calulateDistance();
		println("sorting.....");
		SortUtil.quickSort(distances);
		println("building tree.....");
		startBuild();//build tree
		//calculate points and codes of the original location
		println("add points.....");
		addPointsAndCodes();
	}
	
	//calculate point of the original location
	private void addPointsAndCodes() {
		for (Location lo : allLos) {
			Location cur = lo;
			while (cur.getParent() != null) {
				//子节点公用父节点参数
				lo.addPoint(cur.getParent().getPoint());
				//code决定公式中的正例或反例
				lo.addCode(cur.getCode());
				cur = cur.getParent();
			}
		}
	}

	//计算两点之间的距离，对于某个点只要保留它的最小距离
	private void calulateDistance() {
		for (int i = 0; i < locNum - 1; i++) {
			//fisrt one
			Distance min = new Distance(allLos.get(i), allLos.get(i+1));
			for (int j = i+2; j < locNum; j++) {
				Distance cur = new Distance(allLos.get(i),allLos.get(j));
				if (cur.getDis() < min.getDis()) {
					min = cur;
				}
			}
			distances.add(min);
		}
	}

	private void startBuild() {
		//寻环实现防止越栈
		while (!distances.isEmpty()) {
			if (point%1000 == 0) {
				println("Iteration: " + point);
			}
			Distance min = distances.get(0);
			Location lo1 = min.getLo1();
			Location lo2 = min.getLo2();
			Location node = new Location(-1,//内部节点的id用不着
					(lo1.getLatitude()+lo2.getLatitude())/2,
					(lo1.getLongtitude()+lo2.getLongtitude())/2
					);
			lo1.setParent(node);
			lo2.setParent(node);
			lo1.setCode(0);
			lo2.setCode(1);
			node.setPoint(point++);
			rmDistance(lo1,lo2);//n
			//add new distance and 
			//insert it to proper position
			addDistance(node);//n
		}
		
		//不能使用递归实现，有超栈的危险
		//startBuild();
	}

	//add new distance and resort
	private void addDistance(Location node) {
		//add distance
		//只添加最小的距离
		if (!curLos.isEmpty()) {
			Distance min = new Distance(node, curLos.get(0));
			for (int i = 1; i < curLos.size(); i++) {
				Distance cur = new Distance(node, curLos.get(i));
				if (cur.getDis() < min.getDis()) {
					min = cur;
				}
			}
			
			//插入到适当位置
			boolean inserted = false;
			for (int i = 0; i < distances.size(); i++) {
				if (min.getDis() <= distances.get(i).getDis()) {
					distances.add(i, min);
					inserted =  true;
					break;
				}
			}
			if (!inserted) {//add to the last
				distances.add(min);
			}
		}
		//add node
		curLos.add(node);
	}

	private void rmDistance(Location lo1, Location lo2) {
		//remove location
		curLos.remove(lo1);
		curLos.remove(lo2);
		//remove distance
		for (Distance dis : distances) {
			if (dis.hasLocation(lo1)||dis.hasLocation(lo2)) {
				distances.remove(dis);
			}
		}
	}

	/**
	 * read the locations from the file
	 * @param filePath
	 */
	private void initLocations(String filePath) {
		JSONObject los = JsonUtil.getJOb(filePath);
		@SuppressWarnings("unchecked")
		Iterator<String> keys =  los.keys();
		while (keys.hasNext()) {
			try {
				String key = keys.next();
				JSONObject lo = los.getJSONObject(key);
				int id = Integer.parseInt(key);
				Location loc = new Location(
						id, lo.getDouble(LAT),
						lo.getDouble(LON));
				allLos.add(loc);
				curLos.add(loc);
			} catch (JSONException e) {
				e.printStackTrace();
			}
		}
		locNum = allLos.size();
	}
	
	public LocationData[] getAllLocations() {
		LocationData[] locations = new LocationData[locNum];
		for (int i = 0; i < locations.length; i++) {
			Location pos = allLos.get(i);
			locations[i] = new LocationData(
					pos.getId(),
					pos.getPoints(),
					pos.getCodes());
		}
		return locations;
	}
	
	public static void println(String str) {
		System.out.println(str+ ".Time:" +
				(System.currentTimeMillis()/1000 - startTime) + "s");
	}
}
