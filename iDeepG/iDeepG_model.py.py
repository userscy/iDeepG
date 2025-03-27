import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from cross_attention import MultiHeadAttention
from LucaOne_inference.get_embedding import get_embedding, load_model


def pad_embedding(embedding):  
    target_shape = (1, 2804, 2560)
    padding_size = target_shape[1] - embedding.size(1)
    if padding_size > 0:
        padded_embedding = F.pad(embedding, (0, 0, 0, padding_size, 0, 0))
    return padded_embedding


class MLPClassifier(Module):
    def __init__(
        self,
        hidden_size,
        dropout,
    ):
        super(MLPClassifier, self).__init__()
        
        # Global Average Pooling to reduce the input dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP layers with reduced parameters
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
        # Activation function
        self.gelu = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.global_avg_pool(x.permute(0, 2, 1))
        x = x.view(x.size(0), -1)
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No softmax here, return logits directly
        
        return x

class iDeepGModel(Module):

    def __init__(
        self,
        hidden_size,
        heads,
        omics_of_interest_size,
        other_omic_size,
        dropout,
    ):
        super().__init__()

        self.cross_attention_layer_protein = MultiHeadAttention(
            num_attention_heads=heads,
            attention_head_size=omics_of_interest_size // heads,
            hidden_size=hidden_size,
            attention_probs_dropout_prob=0,
            max_position_embeddings=0,
            omics_of_interest_size=omics_of_interest_size,
            other_omic_size=other_omic_size,
            position_embedding_type="absolute"
        )
        self.mlp = MLPClassifier(hidden_size, dropout)

    def forward(
            self,
            rna_embedding,
            protein_embedding
    ):

        cross_attn_output = self.cross_attention_layer_protein.forward(
            hidden_states=rna_embedding,
            encoder_hidden_states=protein_embedding,
        )
        
        new_rna_embedding = cross_attn_output["embeddings"]
        attention_scores = cross_attn_output["attention_scores"]
        output = self.mlp(new_rna_embedding)

        return torch.softmax(output, dim=-1)
    
    def predict(self, dna_seq, protein_seq):
        
        lucaone_global_log_filepath = './LucaOne_inference/models/llm/logs/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/logs.txt'
        lucaone_global_model_dirpath = './LucaOne_inference/models/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000'
        lucaone_global_args_info, lucaone_global_model_config, lucaone_global_model, lucaone_global_tokenizer = load_model(lucaone_global_log_filepath, lucaone_global_model_dirpath)
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        rna_embedding, processed_seq_len = get_embedding(lucaone_global_args_info,
                                       lucaone_global_model_config,
                                       lucaone_global_tokenizer,
                                       lucaone_global_model,
                                       dna_seq,
                                       'gene',
                                       DEVICE)
        rna_embedding = rna_embedding.hidden_states
        rna_embedding = rna_embedding[0, 1:-1, :]
        rna_embedding = torch.unsqueeze(rna_embedding, dim=0)
        
        protein_embedding, processed_seq_len = get_embedding(lucaone_global_args_info,
                                       lucaone_global_model_config,
                                       lucaone_global_tokenizer,
                                       lucaone_global_model,
                                       protein_seq,
                                       'prot',
                                       DEVICE)
        protein_embedding = protein_embedding.hidden_states_b
        protein_embedding = protein_embedding[0, 1:-1, :]
        protein_embedding = torch.unsqueeze(protein_embedding, dim=0)
        protein_embedding = pad_embedding(protein_embedding)
        
        self.eval()
        with torch.no_grad():
            prob = self.forward(rna_embedding, protein_embedding)
            pred = torch.argmax(prob, dim=-1)
        
        return prob, pred, rna_embedding, protein_embedding   