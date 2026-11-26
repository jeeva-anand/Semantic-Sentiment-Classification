

def word_vector(token, size):
  vec = np.zeros(size).reshape((1, size))
  count = 0
  for x in token:
    try:
      vec += vector_model.wv[x].reshape((1, size))
      count += 1
    except KeyError:
      pass

  if count != 0:
    vec /= count

  return vec

