import random

how_much_to_gen: int = 10
seq_size: int = 7000
len_per_line: int = 70

with open('random_FASTA.fsa', 'w') as f:
    amino_acids: list = \
        ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for i in range(how_much_to_gen):
        f.write('> random ' + str(i) + '\n')
        random_seq: str = ''
        for _ in range(seq_size):
            random_seq += random.choice(amino_acids)
        fasta_seq: list = [str(random_seq[i:i + len_per_line]) + '\n' for i in range(0, seq_size, len_per_line)]
        f.writelines(fasta_seq)
