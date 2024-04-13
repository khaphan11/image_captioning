import torch
from torch.utils.data import Dataset
from pickle import load
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, pad_sequences


class ImageDataset(Dataset):
    def __init__(self,
                 max_length: int,
                 vocab_size: int) -> None:
        """"""
        self.captions = load(open("encoded_captions.pkl", "rb"))
        self.images = load(open("encoded_train_images.pkl", "rb"))
        self.w2i = load(open("w2i.pkl", "rb"))
        self.max_length = max_length
        self.vocab_size = vocab_size


    def __len__(self) -> int:
        """"""
        return len(self.images)


    def __getitem__(self,
                    idx: int) -> set([torch.Tensor, int]):
        """"""
        id_, caps = list(self.captions.items())[idx]
        image = self.images[id_]
        x_image, x_cap, y = [], [], []

        for cap in caps:
            # encode the sequence
            seq = [self.w2i[word] for word in cap.split(' ') if word in self.w2i]

            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]

                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]

            # store
            x_image.append(image)
            x_cap.append(in_seq)
            y.append(out_seq)

        return [torch.Tensor(x_image), torch.Tensor(x_cap).to(torch.int64)], torch.Tensor(y)


    def get_batch(self, batch_size):
        """"""
        X_image, X_cap, y = [], [], []
        n = 0
        while 1:
            for id, caps in self.captions.items():
                n += 1
                image = self.images[id]
                for cap in caps:
                    # encode the sequence
                    seq = [self.w2i[word] for word in cap.split(' ') if word in self.w2i]

                    for i in range(1, len(seq)):
                        # split into input and output pair
                        in_seq, out_seq = seq[:i], seq[i]

                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]

                        # store
                        X_image.append(image)
                        X_cap.append(in_seq)
                        y.append(out_seq)
                if n == batch_size:
                    yield [torch.Tensor(X_image), torch.Tensor(X_cap).to(torch.int64)], torch.Tensor(y)
                    X_image, X_cap, y = [], [], []
                    n = 0
