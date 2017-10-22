import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class Image2CaptionDecoder(chainer.Chain):
    def __init__(self, vocab_size, img_feature_dim=2048, hidden_dim=512, dropout_ratio=0.5, train=True, n_layers=1):
        super(Image2CaptionDecoder, self).__init__(
            embed_word = L.EmbedID(vocab_size, hidden_dim),
            embed_img = L.Linear(img_feature_dim, hidden_dim),
            lstm = L.NStepLSTM(n_layers=n_layers, in_size=hidden_dim, out_size=hidden_dim, dropout=dropout_ratio),
            decode_word = L.Linear(hidden_dim, vocab_size)
        )
        self.dropout_ratio = dropout_ratio
        self.train = train
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def input_cnn_feature(self, hx, cx, img_feature):
        h = self.embed_img(img_feature)
        h = [F.reshape(img_embedding, (1, self.hidden_dim)) for img_embedding in h]
        hy, cy, ys = self.lstm(hx, cx, h)
        return hy, cy

    def __call__(self, hx, cx, caption_batch):
        xs = [self.embed_word(caption) for caption in caption_batch]
        hy, cy, ys = self.lstm(hx, cx, xs)
        predicted_caption_batch = [self.decode_word(generated_caption) for generated_caption in ys]
        if self.train:
            loss=0
            for y, t in zip(predicted_caption_batch, caption_batch):
                #loss+=F.softmax_cross_entropy(y[0:-1], t[1:])
                loss+=F.softmax_cross_entropy(y, t)

            return loss/len(predicted_caption_batch)
        else:
            return hy, cy, predicted_caption_batch

if __name__ == '__main__':
    img_feature = np.zeros([2, 2048], dtype=np.float32)
    x_batch = [[1, 2, 3, 4, 2, 3, 0, 2], [1, 2, 3, 3, 1]]
    x_batch = [np.array(x, dtype=np.int32) for x in x_batch]
    model = Image2CaptionDecoder(5)
    batch_size = len(x_batch)
    hx = np.zeros((model.n_layers, batch_size, model.hidden_dim), dtype=np.float32)
    cx = np.zeros((model.n_layers, batch_size, model.hidden_dim), dtype=np.float32)
    hx, cx = model.input_cnn_feature(hx, cx, img_feature)
    
    with chainer.using_config('train', False):
        loss = model(hx, cx, x_batch)
        print(loss)
