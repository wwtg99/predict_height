import pandas as pd


def process_mafs(filepath):
    rs = []
    mafs = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cols = line.split('\t')
            if len(cols) < 3:
                continue
            elif len(cols) > 3:
                m = cols[3].strip().split(':')
                if len(m) < 3:
                    continue
                maf = m[0]
                maf_allele = m[2]
            else:
                maf = None
                maf_allele = None
            rs.append(cols[0])
            mafs.append([cols[1], cols[2], maf, maf_allele])
    df = pd.DataFrame(mafs, index=rs, columns=['ref', 'alt', 'maf', 'maf_allele'])
    return df


if __name__ == '__main__':
    df = process_mafs('data/maf.tsv')
    df.to_csv('lib/maf.csv')
