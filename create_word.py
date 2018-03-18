import codecs
import collections
from operator import itemgetter
import os

ORIGIN_DIR = os.getcwd() + "/origin_data"
TARGET_DIR = os.getcwd() + "/target_data"
MODE = "TRANSLATE_ZH"    # 将MODE设置为"PTB", "TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "PTB":            
    RAW_DATA = ORIGIN_DIR+"/ptb.train.txt"  
    VOCAB_OUTPUT = TARGET_DIR + "/ptb.vocab"                         
elif MODE == "TRANSLATE_ZH":  
    RAW_DATA = ORIGIN_DIR + "/train.txt.zh.txt"
    VOCAB_OUTPUT = TARGET_DIR + "/zh.vocab"
    VOCAB_SIZE = 4000
elif MODE == "TRANSLATE_EN": 
    RAW_DATA = ORIGIN_DIR + "/train.txt.en.txt"
    VOCAB_OUTPUT = TARGET_DIR + "/en.vocab"
    VOCAB_SIZE = 10000

counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

sorted_word_to_cnt = sorted(
    counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

if MODE == "PTB":
    sorted_words = ["<eos>"] + sorted_words
elif MODE in ["TRANSLATE_EN", "TRANSLATE_ZH"]:
    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
    if len(sorted_words) > VOCAB_SIZE:
        sorted_words = sorted_words[:VOCAB_SIZE]

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")