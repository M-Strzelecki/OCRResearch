# example of basic string matching (currently works only for same length string)

# two string to compare
seq1 = 'atgcttcggcaagactcaaaaaaata'
seq2 = 'atccttcggcaagactccaaaaaata'

#zip function to pair the two string (coresponding characters) as tuple
zip_seqs = zip(seq1,seq2)
# print(list(zip_seqs))

#enumeratae to track index
enum_seqs = enumerate(zip_seqs)
# print(list(enum_seqs))

# printout index where miss match occures
for i, (a,b) in enum_seqs:
    if a != b:
        print(f'index: {i}')