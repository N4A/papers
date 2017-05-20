import operator
import doc_manager as dm
import json


class Node(object):

    def __init__(self, word, value):
        self.value = value
        self.word = word
        self.parent = None
        self.code = 0
        self.point = 0
        self.points_reverse = []
        self.codes_reverse = []


class Huffman(object):

    def __init__(self):
        self.all_words = []
        self.cur_words = []
        self.point = 0

    def build_tree(self, all_words):
        sorted_words = sorted(all_words.items(), key=operator.itemgetter(1))
        self.cur_words = list(map(lambda word: Node(word[0], word[1]), sorted_words))
        for word in self.cur_words:
            self.all_words.append(word)

        # start build tree
        while len(self.cur_words) > 1:
            if self.point % 1000 == 0:
                print('Iteration: %s' % self.point)
                print('Len of left Words: %s' % len(self.cur_words))

            min1 = self.cur_words[0]
            min2 = self.cur_words[1]
            parent = Node('', min1.value + min2.value)
            min1.parent = parent
            min2.parent = parent
            # left branch
            min1.code = 0
            # right branch
            min2.code = 1
            # parameter index of parent
            parent.point = self.point
            self.point += 1

            # remove old words
            self.cur_words.remove(min1)
            self.cur_words.remove(min2)

            # insert new words
            added = 0
            for i in range(len(self.cur_words)):
                if parent.value < self.cur_words[i].value:
                    self.cur_words.insert(i, parent)
                    added = 1
                    break
            if added == 0:
                self.cur_words.append(parent)

    def calculate_points_codes(self):
        for word in self.all_words:
            cur = word
            while cur.parent:
                word.points_reverse.append(cur.parent.point)
                word.codes_reverse.append(cur.code)
                cur = cur.parent


def get_node_data(node):
    return {'points': node.points_reverse[::-1], 'codes': node.codes_reverse[::-1]}


def test():
    all_words, all_docs = dm.handle_raw_data('data/movielen/ml_plot.txt')
    huffman = Huffman()
    huffman.build_tree(all_words)
    huffman.calculate_points_codes()
    huffman_tree = {}
    for node in huffman.all_words:
        huffman_tree[node.word] = get_node_data(node)
    plot = open('data/movielen/handled/tree.txt', 'w')
    plot.write(json.dumps(huffman_tree))
    doc = open('data/movielen/handled/doc.txt', 'w')
    doc.write(json.dumps(all_docs))
    words = open('data/movielen/handled/words.txt', 'w')
    words.write(json.dumps(all_words))

    plot.close()
    doc.close()
    words.close()

if __name__ == '__main__':
    test()
