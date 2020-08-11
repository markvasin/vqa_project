import json

# json_file = '../dataset/dicts.json'
# token_to_ix, max_token = json.load(open(json_file, 'r'))[2:]
# f = open("../dataset/gqa_vocab.txt", "w")
# for word in token_to_ix:
#     f.write(word + '\n')
# f.close()

json_file = '../dataset/dicts.json'
ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))[:2]
f = open("../dataset/gqa_answers.txt", "w")
for word in ans_to_ix:
    f.write(word + '\n')
f.close()