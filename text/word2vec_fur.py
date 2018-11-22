from gensim.models import Word2Vec

model = Word2Vec.load("model/word2vec_wx/word2vec_wx")

# 继续训练word2vec
sentences = []
with open("data/samples/positive.txt", encoding='utf8') as f:
    for line in f:
        words = [w for w in line.split() if w in model]
        if not words:
            continue
        sentences.append(words)
with open("data/samples/negative.txt", encoding='utf8') as f:
    for line in f:
        words = [w for w in line.split() if w in model]
        if not words:
            continue
        sentences.append(words)

model.train(sentences, epochs=1, total_examples=len(sentences))
model.save("model/word2vec/word2vec")