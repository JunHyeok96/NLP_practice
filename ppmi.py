from common.util import *
import numpy as np

text = 'you say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision = 3)

print('동시발생 행렬')
print(C)
print('-'*50)
print('PPMI')
print(W)