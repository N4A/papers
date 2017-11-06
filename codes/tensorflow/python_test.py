#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/13 13:51
# @Author  : duocai


def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    reverse = {}
    length = len(nums)
    for i in range(length):
        key = target - nums[i]
        if key in reverse.keys():
            return [i, reverse[key]]
        reverse[nums[i]] = i


def lengthRecursion(s):
    """
        :type s: str
        :rtype: str
        """
    if len(s) <= 1:
        return s

    sub = lengthRecursion(s[1:])
    if sub == s[1:len(sub) + 1] and sub.find(s[0]) < 0:
        return s[0] + sub
    else:
        sub1 = s[0]
        i = 1
        while True:
            if sub1.find(s[i]) < 0:
                sub1 += s[i]
                i += 1
            else:
                break
        if len(sub) > len(sub1):
            return sub
        return sub1


def lengthOfLongestSubstring(s):
    """
    :type s: str
    :rtype: str
    """
    length = len(s)
    if length <= 1:
        return length
    longest = s[length - 1]
    i = length - 2
    while i >= 0:
        if s[i + 1:len(longest) + i + 1] == longest and longest.find(s[i]) < 0:
            longest = s[i] + longest
        else:
            sub = s[i]
            j = i + 1
            while True:
                if sub.find(s[j]) < 0:
                    sub += s[j]
                    j += 1
                else:
                    break
            if len(sub) >= len(longest):
                longest = sub
        i -= 1
    return longest


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def CMBT(nums):
    """
    :type nums: List[int]
    :rtype: TreeNode
    """
    heap_size = len(nums)

    def lc(i):
        return 2 * i + 1

    def rc(i):
        return 2 * i + 2

    def exchange(a, i, j):
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

    def max_heapify(a, i):
        l = lc(i)
        r = rc(i)
        lgt = i
        if l < heap_size and a[l] > a[lgt]:
            lgt = l
        if r < heap_size and a[r] > a[lgt]:
            lgt = r
        if lgt != i:
            exchange(a, i, lgt)
            max_heapify(a, lgt)

    def build_max_heap(a):
        i = int(heap_size / 2)
        while i >= 0:
            max_heapify(a, i)
            i -= 1

    def build_tree(node, i):
        l = lc(i)
        r = rc(i)
        if l < heap_size:
            left_node = TreeNode(nums[l])
            node.left = left_node
            build_tree(left_node, l)

        if r < heap_size:
            right_node = TreeNode(nums[r])
            node.right = right_node
            build_tree(right_node, r)

    build_max_heap(nums)
    root = TreeNode(nums[0])
    build_tree(root, 0)

    return root


if __name__ == '__main__':
    print(CMBT([3, 2, 1, 6, 0, 5]))
