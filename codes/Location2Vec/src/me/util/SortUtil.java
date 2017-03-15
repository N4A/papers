package me.util;

import java.util.List;

import me.tree.Distance;

/**
 * @author Duocai Wu
 * @date 2017年1月9日
 * @time 下午3:27:32
 *
 */
public class SortUtil {
    /**
     * 快速排序
     * @param n - Distance 数组
     */
    public static void quickSort(List<Distance> n) {
        if (isEmpty(n))
            return;
        quickSort(n, 0, n.size() - 1);
    }
    
    
	private static void quickSort(List<Distance> n, int l, int h) {
        if (isEmpty(n))
            return;
        if (l < h) {
            int pivot = partion(n, l, h);
            quickSort(n, l, pivot - 1);
            quickSort(n, pivot + 1, h);
        }
    }
	
    private static int partion(List<Distance> n, int start, int end) {
        Distance tmp = n.get(start);
        while (start < end) {
            while (n.get(end).getDis() >= tmp.getDis() && start < end)
                end--;
            if (start < end) {
            	n.set(start++, n.get(end));
            }
            while (n.get(start).getDis() < tmp.getDis() && start < end)
                start++;
            if (start < end) {
            	n.set(end--, n.get(start));
            }
        }
        n.set(start, tmp);
        return start;
    }
    
    private static boolean isEmpty(List<Distance> n) {
        return n == null || n.size() == 0;
    }
}
