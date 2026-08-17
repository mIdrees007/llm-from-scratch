import torch.nn as nn


class FeedForward(nn.Module):
    """ 
    1.5 FFN with expansion factor mult
    Dimension: 
    
    inputL (B, T, d_model)
    inner: (B, T, mult*d_model)
    outputL (B, T, d_model)
    
    
    mult*d_model means the hidden width is mult times larger than d_model 
    
    Typical values : mult=4 for GELU FFN GPT - style blocks
    
    """
    
    
    
    def __init__(self, d_model: int, mult: int =4, dropout: float = 0.0):
        
        super().__init__()
        self.net = nn.Sequential(
            
            
            
            nn.Linear(d_model, mult * d_model),
            nn.GELU(), 
            nn.Linear(mult *  d_model, d_model), 
            nn.Dropout(dropout),
            
        )
        
    def forward(self, x):
        return self.net(x)
    
    