import os
import torch
from iDeepG import iDeepGModel
import argparse
from Bio import SeqIO

def parse_args():
    parser = argparse.ArgumentParser(description='iDeepG model prediction')
    parser.add_argument('--rna_seq_path', type=str, required=True, help='Path to RNA sequences FASTA file')
    parser.add_argument('--prot_seq_path', type=str, required=True, help='Path to protein sequence FASTA file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save output')
    parser.add_argument('--model_weights', type=str, default='./val_model_epoch_12.pth', 
                        help='Path to model weights file')
    return parser.parse_args()

def load_sequences(rna_path, prot_path):
    # Load RNA sequences
    rna_seqs = []
    with open(rna_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            rna_seqs.append(str(record.seq))
    
    # Load protein sequence (only the first one)
    with open(prot_path) as handle:
        protein_seq = str(next(SeqIO.parse(handle, "fasta")).seq)
    
    return rna_seqs, protein_seq

def main():
    args = parse_args()
    
    # Create save directory if not exists
    os.makedirs(args.save_path, exist_ok=True)
    
    # Load sequences
    rna_seqs, protein_seq = load_sequences(args.rna_seq_path, args.prot_seq_path)
    
    # Initialize device
    device = torch.device(args.device)
    
    # Load model
    model = iDeepGModel(hidden_size=2560, heads=8,
                       omics_of_interest_size=2560,
                       other_omic_size=2560, dropout=0.3).to(device)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.eval()
    
    # Prepare output file
    output_file = os.path.join(args.save_path, 'predictions.txt')
    
    with open(output_file, 'w') as f_out, torch.no_grad():
        f_out.write("RNA_ID\tProbability\n")  # Header
        
        # Process each RNA sequence
        for i, rna_seq in enumerate(rna_seqs):
            prob, pred, _, _ = model.predict(rna_seq, protein_seq)
            f_out.write(f"RNA_{i+1}\t{prob.item()}\n")
    
    print(f"Predictions saved to {output_file}")

if __name__ == '__main__':
    main()
