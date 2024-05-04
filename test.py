import math
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

model_fname = "C:/Users/강평종/Desktop/딥러닝/soyword.model"

word_extractor = WordExtractor(min_frequency=100, min_cohesion_forward=0.05, min_right_branching_entropy=0.0)
word_extractor.load(model_fname)
scores = word_extractor.word_scores()
scores = {key:(scores[key].cohesion_forward * math.exp(scores[key].right_branching_entropy)) for key in scores.keys()}
tokenizer = LTokenizer(scores=scores)
tokens = tokenizer.tokenize("5월 4일 토요일 아침, 강평종이 지나가던 행인에게 폭력을 행사했다.")

print(tokens)