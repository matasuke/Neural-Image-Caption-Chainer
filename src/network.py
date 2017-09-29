import chainer
from chainer import functions as F
from chainer import links as L


class ImageCaption(chainer.Chain):
    def __init__(self, vocab_size, img_feature_dim = 2048, hidden_dim = 512, dropout_ratio = 0.5, train=True):
        super(ImageCaption, self).__init__(
                word_embed = L.EmbedID(vocab_size, hidden_dim),
                img_vec = L.Linear(img_feature_dim, hidden_dim),
                lstm = L.LSTM(hidden_dim, hidden_dim),
                word_decode = L.Linear(hidden_dim, vocab_size)
        )

