from text2vec import SentenceModel


# sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
# sentences are from '/Users/zihanwu/codes/NLP/raw_chat_corpus/tieba-305w/tieba.dialogues"
# example: 
# 24直播网有免费高清的	有的
# 前排，鲁迷们都起床了吧	标题说助攻，但是看了那球，真是活生生的讽刺

def read_sentences(file_path, max_lines=10000):
    sentences = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i > max_lines:
                break
            for sentence in line.split('\t'):
                sentences.append(sentence.strip())
    return sentences


sentences = read_sentences('/Users/zihanwu/codes/NLP/raw_chat_corpus/tieba-305w/tieba.dialogues')

model = SentenceModel('shibing624/text2vec-base-chinese')
embeddings = model.encode(sentences)
print(embeddings)
