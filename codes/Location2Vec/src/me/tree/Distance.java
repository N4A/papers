package me.tree;

/**
 * @author Duocai Wu
 * @date 2017年1月9日
 * @time 下午3:24:38
 *
 */
public class Distance {
	private Location lo1;
	private Location lo2;
	private double dis;

	/**
	 * @param lo1
	 * @param lo2
	 */
	public Distance(Location lo1, Location lo2) {
		this.lo1 = lo1;
		this.lo2 = lo2;
		this.dis = Location.Distance(lo1, lo2);
	}

	/**
	 * compare the value of distance
	 * @param o - the distance compares to
	 * @return
	 */
	public boolean lessThen(Distance o) {
		return this.dis < o.getDis();
	}

	public double getDis() {
		return dis;
	}
	
	public Location getLo1() {
		return lo1;
	}

	public Location getLo2() {
		return lo2;
	}

	public boolean hasLocation(Location lo) {
		return lo == lo1 || lo == lo2;
	}
}
