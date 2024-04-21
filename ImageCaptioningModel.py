import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from pickle import load


class ImageCaptioningModel(nn.Module):
    def __init__(self,
                 image_shape: tuple,
                 max_length: int,
                 vocab_size: int,
                 embedding_dim: int) -> None:
        """"""
        super().__init__()
        self.image_shape = image_shape
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        embedding_matrix = load(open("embedding_matrix.pkl", "rb"))


        self.fe1 = nn.Dropout(0.5)
        self.fe2 = nn.Linear(image_shape[0], 256)

        self.se1 = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.se1.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).to(torch.float))
        for param in self.se1.parameters():
            param.requires_grad = False
        self.se2 = nn.Dropout(0.5)
        self.se3 = nn.LSTM(embedding_dim, 256)

        # decoder1 = fe2 + se3
        self.decoder2 = nn.Linear(256, 256)

        self.outputs = nn.Linear(256, vocab_size)


    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """"""
        image_input = torch.Tensor(inputs[0])
        cap_input = torch.Tensor(inputs[1])

        fe1 = self.fe1(image_input)
        fe2 = F.relu(self.fe2(fe1))

        se1 = self.se1(cap_input)
        se2 = self.se2(se1)
        se3, (_, _) = self.se3(se2)
        se3 = se3[:, -1, :]


        # if len(se3.shape) == 2:
        #     decoder1 = torch.stack([torch.Tensor(se + fe) for se, fe in zip(se3, fe2)])
        # else:
        decoder1 = fe2 + se3

        decoder2 = self.decoder2(decoder1)
        output = self.outputs(decoder2)
        output = F.softmax(output, dim=1)

        return output


    def summary(self) -> None:
        """"""
        summary(self.model, [self.image_shape, (self.max_length, )])