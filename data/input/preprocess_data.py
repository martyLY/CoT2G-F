import random
import pandas as pd
import numpy as np

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--fasta-filename', type=str, default=None,
                        help='t time slice spike fasta file name')
    parser.add_argument('--neighbor-fasta-filename', type=str, default=None,
                        help='t-1 time slice spike fasta file name')
    parser.add_argument('--neighbor-metadata-filename', type=str, default=None,
                        help='the neighbor metadata filename, the column names are id and t_1, id is t time slice spike id and t_1 is  t-1 time slice spike id')
    parser.add_argument('--split', type=bool, default=None,
                        help='set to True to split spike to s1 and s2 to introduce neighbor sequence')
    parser.add_argument('--wo-neighbor', type=bool, default=None,
                        help='set to True to split spike to s1 and s2 and not to introduce neighbor sequence')
    parser.add_argument('--mask', type=bool, default=None,
                        help='set to True to randomly continuously mask amino acids')
    parser.add_argument('--s1-filename', type=str, default='s1_seq_to_seq_demo.csv',
                        help='s1 spike preprocess results file name')
    parser.add_argument('--s2-filename', type=str, default='s2_seq_to_seq_demo.csv',
                        help='s2 spike preprocess results file name')
    parser.add_argument('--s-filename', type=str, default='s_seq_to_seq_demo.csv',
                        help='spike preprocess results file name')
    
    args = parser.parse_args()
    return args


def read_fasta(fasta_name):
    data = {'id': [], 'sequence': []}
    with open(fasta_name, 'r') as f:
        seq = ''
        key = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                if key:
                    data['id'].append(key)
                    data['sequence'].append(seq)
                    seq = ''
                key = line[1:]
            else:
                seq += line
        if key:
            data['id'].append(key)
            data['sequence'].append(seq)

    data_df = pd.DataFrame(data)
    return data_df

def get_mask_index(n):
    mask_num= int(n*0.15)
    idx=np.arange(0,n).tolist()
    remain=idx
    mask=[]
    tri_mask_num = random.randrange(0, int(mask_num/3)-1)
    i=0
    while i <tri_mask_num:
        x=random.choice(remain)
        if (x in remain) and (x+1 in remain)and (x+2 in remain):
            remain.remove(x)
            remain.remove(x+1)
            remain.remove(x+2)
            mask.append(x)
            mask.append(x+1)
            mask.append(x+2)
            i+=1
    double_mask_num = random.randrange(0, int((mask_num-len(mask))/2))
    i=0
    while i <double_mask_num:
        x=random.choice(remain)
        if (x in remain) and (x+1 in remain):
            remain.remove(x)
            remain.remove(x+1)
            mask.append(x)
            mask.append(x+1)
            i+=1
    while len(mask) <mask_num:
        x=random.choice(remain)
        if (x in remain) :
            remain.remove(x)
            mask.append(x)
            i+=1
    return mask


def main():
    args = parse_args()
    print('load {} data...'.format(args.fasta_filename))
    data_df = read_fasta(args.fasta_filename)

    if args.split:
        print('load neighbor(t-1) {} data && join to t data...'.format(args.neighbor_fasta_filename))
        neighbor_fasta_df = read_fasta(args.neighbor_fasta_filename)
        neighbor_fasta_df = neighbor_fasta_df.rename(columns = {'id':'nearest_neighbor_t-1', 'sequence':'neighbor_sequence'})
        neighbor_df =pd.read_csv(args.neighbor_metadata_filename,sep=' ')
        step_one_df = pd.merge(data_df, neighbor_df, how='inner',on='id')
        step_two_df = pd.merge(step_one_df, neighbor_fasta_df, how='inner',left_on='t_1' ,right_on='nearest_neighbor_t-1')
        input_df = step_two_df.drop_duplicates(keep='first')
    else:
        input_df = data_df
    
    input_df = input_df.drop_duplicates(subset=['sequence'])

    if args.mask:
        print('randomly continuous mask amino acids...')
        sequence_list = input_df['sequence'].to_list()
        input_df = input_df.drop(columns=['sequence'])
        mask_sequence_list = []
        n = len(sequence_list)
        for s in sequence_list:
            mask = get_mask_index(len(s))
            mask_s=[s[i] if i not in mask else '-' for i in range(len(s))]
            mask_s=''.join(mask_s)
            mask_sequence_list.append(mask_s)
        input_df['sequence'] = mask_sequence_list
        
    def get_s1(s):
        return s[:688]
    def get_s2(s):
        return s[688:]

    input_df['s1'] = input_df['sequence'].apply(get_s1)
    input_df['s2'] = input_df['sequence'].apply(get_s2)

    if args.split:
        input_df['neighbor_s1'] = input_df['neighbor_sequence'].apply(get_s1)
        input_df['neighbor_s2'] = input_df['neighbor_sequence'].apply(get_s2)
        input_df['s1_input'] = input_df['s1'] + ' ' + input_df['neighbor_s1']
        input_df['s1_output'] = ' '
        input_df[['s1_input', 's1_output']].to_csv(args.s1_filename, index=0, header=['input','output'])
        input_df['s2_input'] = input_df['s2'] + ' ' + input_df['neighbor_s2']
        input_df['s2_output'] = ' '
        input_df[['s2_input', 's2_output']].to_csv(args.s2_filename, index=0, header=['input','output'])
        print('s1 spike w/ neighbor(t-1) output to {}.\n'.format(args.s1_filename), \
            's2 spike w/ neighbor(t-1) output to {}.'.format(args.s2_filename))

    if args.wo_neighbor:
        input_df['s1_output'] = ' '
        input_df[['s1', 's1_output']].to_csv(args.s1_filename, index=0, header=['input','output'])
        input_df['s2_output'] = ' '
        input_df[['s2', 's2_output']].to_csv(args.s2_filename, index=0, header=['input','output'])
        print('s1 spike w/o neighbor(t-1) output to {}.\n'.format(args.s1_filename), \
            's2 spike w/o neighbor(t-1) output to {}.'.format(args.s2_filename))


    input_df['output'] = ' '
    input_df[['sequence', 'output']].to_csv(args.s_filename, index=0, header=['input','output'])
    print('s spike w/o neighbor(t-1) output to {}'.format(args.s_filename))


if __name__ == "__main__":
    main()