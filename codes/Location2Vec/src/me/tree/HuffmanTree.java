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
 * @date 2017��1��9��
 * @time ����1:18:50
 *
 */
public class HuffmanTree {
	private final String LAT = "latitude";
	private final String LON = "longitude";
	
	private int locNum; //�ص���Ŀ
	private List<Location> allLos = new ArrayList<>();//ԭʼ���еص�
	
	private List<Location> curLos = new ArrayList<>();//�������Ĺ��������еص�
	private List<Distance> distances = new CopyOnWriteArrayList<>();//�ص�֮��ľ���
	private int point = 0;
	
	private static long startTime;
	
	public HuffmanTree() {
		startTime = System.currentTimeMillis()/1000;
	}
	
	
	/**
	 * build huffman tree according to the location in the file
	 * ������֦��0��ǣ���֦��1���
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
				//�ӽڵ㹫�ø��ڵ����
				lo.addPoint(cur.getParent().getPoint());
				//code������ʽ�е���������
				lo.addCode(cur.getCode());
				cur = cur.getParent();
			}
		}
	}

	//��������֮��ľ��룬����ĳ����ֻҪ����������С����
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
		//Ѱ��ʵ�ַ�ֹԽջ
		while (!distances.isEmpty()) {
			if (point%1000 == 0) {
				println("Iteration: " + point);
			}
			Distance min = distances.get(0);
			Location lo1 = min.getLo1();
			Location lo2 = min.getLo2();
			Location node = new Location(-1,//�ڲ��ڵ��id�ò���
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
		
		//����ʹ�õݹ�ʵ�֣��г�ջ��Σ��
		//startBuild();
	}

	//add new distance and resort
	private void addDistance(Location node) {
		//add distance
		//ֻ�����С�ľ���
		if (!curLos.isEmpty()) {
			Distance min = new Distance(node, curLos.get(0));
			for (int i = 1; i < curLos.size(); i++) {
				Distance cur = new Distance(node, curLos.get(i));
				if (cur.getDis() < min.getDis()) {
					min = cur;
				}
			}
			
			//���뵽�ʵ�λ��
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
