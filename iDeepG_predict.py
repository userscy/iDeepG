import os
import torch
from iDeepG import iDeepGModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model

iDeepG_model_weights = './val_model_epoch_12.pth'
model = iDeepGModel(hidden_size=2560, heads=8,
                     omics_of_interest_size=2560,
                     other_omic_size=2560, dropout=0.3).to(DEVICE)
model.load_state_dict(torch.load(iDeepG_model_weights))
model.eval()


# Predict seq

RNA_seq = 'AAGCAAAAAUCUGCCUUGAGAUCAUGCAGAGAACUGGUGCUCACUUGGAGCUGUCUUUGGCCAAAGACCAAGGCCUCUCCAUCAUGGUGUCAGGAAAGCUG'
Protein_seq = 'MAEGGASKGGGEEPGKLPEPAEEESQVLRGTGHCKWFNVRMGFGFISMINREGSPLDIPVDVFVHQSKLFMEGFRSLKEGEPVEFTFKKSSKGLESIRVTGPGGSPCLGSERRPKGKTLQKRKPKGDRCYNCGGLDHHAKECSLPPQPKKCHYCQSIMHMVANCPHKNVAQPPASSQGRQEAESQPCTSTLPREVGGGHGCTSPPFPQEARAEISERSGRSPQEASSTKSSIAPEEQSKKGPSVQKRKKT'

with torch.no_grad():
    prob, pred, rna_embedding, protein_embedding = model.predict(RNA_seq, Protein_seq)
print("Prob:", prob)
print("Label prediction:", pred.item())
