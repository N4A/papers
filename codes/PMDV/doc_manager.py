import json


def handle_raw_data(path):
    all_words = {}
    all_docs = {}
    f = open(path, 'r')
    # analyze the docs
    for line in f:
        # split line to id and text
        temp = line.split('::')

        # ignore some irregular texts
        if len(temp) != 2:
            continue

        # get id
        id = temp[0]
        # get text
        docs = temp[1].replace('\n', '').split('|')
        if docs[-1] == '':
            docs = docs[0:len(docs)-1]
        all_docs[id] = docs

        # calculate the frequency of words
        for doc in docs:
            words = doc.split('\t')
            for word in words:
                # ignore bad words
                if word == '' or word == ' ' or word == '\t':
                    continue
                if word in all_words.keys():
                    all_words[word] += 1
                else:
                    all_words[word] = 1

    return all_words, all_docs


def read_handled_data(path):
    f = open(path, 'r')
    return json.load(f)
