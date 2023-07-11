fasta_filename=FASTA
neighbor_fasta_filename=NEIGHBPR_FASTA
neighbor-metadata-filename=METADATA
s1_filename=S1
s2_filename=S2
s_filename=S

# python preprocess_data.py \
#     --fasta-filename ${fasta_filename} \
#     --neighbor-fasta-filename ${neighbor_fasta_filename} \
#     --neighbor-metadata-filename ${neighbor_metadata_filename} \
#     --split True \
#     --mask True \
#     --s1-filename ${s1_filename} \
#     --s2-filename ${s2_filename} \
#     --s-filename ${s_filename}

# demo for using neighbor_file
# python preprocess_data.py \
#     --fasta-filename S_t_negihbor.fa \
#     --neighbor-fasta-filename S_t_1_neighbor.fa\
#     --neighbor-metadata-filename t_1_neighbor.csv \
#     --split True \
#     --mask True \
#     --s1-filename s1.csv \
#     --s2-filename s2.csv \
#     --s-filename s.csv

python preprocess_data.py \
    --fasta-filename ${fasta_filename} \
    --mask True \
    --wo-neighbor True \
    --s1-filename ${s1_filename} \
    --s2-filename ${s2_filename} \
    --s-filename ${s_filename}
