from torch import nn

class TextClassifier(nn.Module):
    def __init__(self, vocable_size, embedding_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocable_size, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
        
        self.reLU = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc =  nn.Linear(64, num_classes)
        
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.reLU(x)
        x = self.pool(x)
        x = x.squeeze(2)
        x = self.fc(x)
        return x
    
