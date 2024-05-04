from soynlp.word import WordExtractor

corpus_fname = "C:/Users/강평종/Desktop/딥러닝/case1000.txt"
model_fname = "C:/Users/강평종/Desktop/딥러닝/soyword.model"

sentences = [sent.strip() for sent in open(corpus_fname, 'r', encoding='utf-8').readlines()]
# 최소한 100번 등장한 단어만 포함
word_extractor = WordExtractor(min_frequency=100, min_cohesion_forward=0.05, min_right_branching_entropy=0.0)
word_extractor.train(sentences)
word_extractor.save(model_fname)