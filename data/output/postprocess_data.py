import pandas as pd
from pandas import DataFrame

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Post process the csc score and grammer of generated spike protein sequence, only for CoT2G-F method.')
    parser.add_argument('--s1-filename', type=str, default=None,
                        help='s1 generation results file name.')
    parser.add_argument('--s2-filename', type=str, default=None,
                        help='s2 generation results file name.')
    parser.add_argument('--s-results-filename', type=str, default=None,
                        help='s1 and s2 merged results file name.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.s1_filename is None:
        print("Please indict s1 protain generation file name !!")
        return
    elif args.s2_filename is None:
        print("Please indict s2 protain generation file name !!")
        return
    s1_df = pd.read_csv(args.s1_filename)
    s2_df = pd.read_csv(args.s2_filename)
    if s1_df.shape[0] != s2_df.shape[0]:
        print("The count of generated s1 protain and generated s2 protain is suposed to be equal !!")
        return
    s = {'pred_sequences':[],'pred_mutations':[],'csc':[],\
        'grammer':[]}
    s_df = DataFrame(s)
    s_df['pred_sequences'] = s1_df['pred_sequences'] + s2_df['pred_sequences']
    s_df['pred_mutations'] = s1_df['pred_mutations'] +' '+ s2_df['pred_mutations']
    s_df['csc'] = (s1_df['csc'] + s2_df['csc']) / 2
    s_df['grammer'] = (s1_df['grammer'] + s2_df['grammer']) / 2
    s_df.to_csv(args.s_results_filename, index=0)
    print("The merge generated results are saved into {}.".format(args.s_results_filename))

if __name__ == "__main__":
    main()