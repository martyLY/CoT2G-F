import pandas as pd

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess of spike protein sequence data.')
    parser.add_argument('--fasta-filename', type=str, default=None,
                        help='spike protein sequence file name')
    parser.add_argument('--neighbor-metadata-filename', type=str, default=None,
                        help='spike protein sequence neighbor sequence id in last mouth.')
    parser.add_argument('--date-metadata-filename', type=str, default=None,
                        help='the date of spike protein sequence, granularity to month')
    parser.add_argument('--split', type=bool, default=None,
                        help='whether split the spike protein sequence to s1 and s2, CoT2G-F need to set True.')
    parser.add_argument('--s1-input-filename', type=str, default='s1_seq_to_seq_demo.csv',
                        help='preprocessing result output , s1')
    parser.add_argument('--s2-input-filename', type=str, default='s2_seq_to_seq_demo.csv',
                        help='preprocessing result output , s2')
    parser.add_argument('--s-input-filename', type=str, default='s_seq_to_seq_demo.csv',
                        help='preprocessing result output , s, used for Vallina Transformer method.')
    
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

def main():
    args = parse_args()
    data_df = read_fasta(args.fasta_filename)

    if args.split:
        date_df = pd.read_csv(args.date_metadata_filename)
        nei_df = pd.read_csv(args.neighbor_metadata_filename)

        data_df = pd.merge(data_df, date_df, how='inner', on='id')
        step_one_df = pd.merge(data_df, nei_df, how='inner',on='id')
        data_df = data_df.rename(columns = {'id':'nearest_neighbor_t-1', 'sequence':'neighbor_sequence', \
            'date': 'neighbor_date'})
        step_two_df = pd.merge(step_one_df, data_df, how='inner', on='nearest_neighbor_t-1')
        input_df = step_two_df.drop_duplicates(keep='first')
        input_df = input_df.drop(columns=['date', 'neighbor_date'])
    else:
        input_df = data_df

    def get_s1(s):
        return s[:688]
    def get_s2(s):
        return s[688:]

    if args.split:
        input_df['s1'] = input_df['sequence'].apply(get_s1)
        input_df['s2'] = input_df['sequence'].apply(get_s2)
        input_df['neighbor_s1'] = input_df['neighbor_sequence'].apply(get_s1)
        input_df['neighbor_s2'] = input_df['neighbor_sequence'].apply(get_s2)
        input_df['s1_input'] = input_df['s1'] + ' ' + input_df['neighbor_s1']
        input_df['s1_output'] = ' '
        input_df[['s1_input', 's1_output']].to_csv(args.s1_input_filename, index=0, header=['input','output'])
        input_df['s2_input'] = input_df['s2'] + ' ' + input_df['neighbor_s2']
        input_df['s2_output'] = ' '
        input_df[['s2_input', 's2_output']].to_csv(args.s2_input_filename, index=0, header=['input','output'])

    input_df['output'] = ' '
    input_df[['sequence', 'output']].to_csv(args.s_input_filename, index=0, header=['input','output'])

if __name__ == "__main__":
    main()