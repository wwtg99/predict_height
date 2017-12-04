import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder


def load_mafs(filepath):
    """
    Load maf DateFrame
    :param filepath:
    :return:
    """
    return pd.read_csv(filepath, index_col=0)


def process_genotypes(filepath, snp_maf, snp_list=None, **kwargs):
    """
    Process genotype file.

    :param filepath:
    :param snp_maf:
    :param snp_list: get specified snp if provided
    :param bool genotype_label: True if first column is the label of specimen, default False
    :param bool skip_none_rs: True if skip None genotype, default True
    :param bool fill_none: True if auto fill None genotype with most frequent genotype by MAF, default True
    :return:
    """
    conf = dict({
        'genotype_label': False,
        'skip_none_rs': True
    }, **kwargs)
    with open(filepath, encoding='utf-8') as fh:
        if conf['genotype_label']:
            df = genotype_with_label(fh, snp_maf=snp_maf, snp_list=snp_list, **conf)
        else:
            df = genotype_without_label(fh, snp_maf=snp_maf, snp_list=snp_list, **conf)
        return df


def genotype_with_label(fp, snp_maf, snp_list=None, **kwargs):
    """
    Process genotype file with labeled accession.

    :param fp:
    :param snp_maf:
    :param snp_list:
    :param kwargs:
    :return:
    :rtype: pd.DateFrame
    """
    ss = {}
    for line in fp:
        line = line.strip()
        if not line:
            continue
        cols = line.split('\t')  # accession rs allele1 allele2
        if len(cols) > 3:
            acc, rs, a1, a2 = cols[0:4]
            # transform genotype
            if rs in snp_maf.index:
                gt = transform_genotype(snp_maf.loc[rs], a1, a2)
            else:
                if kwargs['skip_none_rs']:
                    continue
                gt = None
            # check snp list
            if snp_list and rs not in snp_list:
                continue
            if rs in ss:
                ss[rs][acc] = gt
            else:
                ss[rs] = {acc: gt}
    return pd.DataFrame(ss)


def genotype_without_label(fp, snp_maf, snp_list=None, **kwargs):
    """
    Process genotype file with one accession.

    :param fp:
    :param snp_maf:
    :param snp_list:
    :param kwargs:
    :return:
    :rtype: pd.DateFrame
    """
    ss = {}
    for line in fp:
        line = line.strip()
        if not line:
            continue
        cols = line.split('\t')  # rs allele1 allele2
        if len(cols) > 2:
            rs, a1, a2 = cols[0:3]
            # transform genotype
            if rs in snp_maf.index:
                gt = transform_genotype(snp_maf.loc[rs], a1, a2)
            else:
                if kwargs['skip_none_rs']:
                    continue
                gt = None
            # check snp list
            if snp_list and rs not in snp_list:
                continue
            ss[rs] = gt
    return pd.DataFrame(ss, index=['genotype'])


def transform_genotype(maf, allele1, allele2):
    """
    Transform genotype to int.
    ref/ref to 0, ref/alt to 1, alt/alt to 2

    :param maf:
    :param allele1:
    :param allele2:
    :return:
    """
    if allele1 is None or allele2 is None:
        # genotype is invalid
        return np.NaN
    else:
        ref = maf['ref']
        alt = maf['alt']
        if allele1 == ref and allele2 == ref:
            return 0
        elif allele1 == alt and allele2 == alt:
            return 2
        else:
            return 1


def get_most_freq_genotype(maf, no_none=False):
    """
    Get highest frequent genotype.

    :param maf:
    :param bool no_none: do not return none
    :return:
    """
    freq = maf['maf']
    allele = maf['maf_allele']
    if freq and allele:
        freq = float(freq)
        if allele == 'ref':
            ref_freq = freq * freq
            alt_freq = (1 - freq) * (1 - freq)
            ref_alt_freq = 1 - ref_freq - alt_freq
        else:
            alt_freq = freq * freq
            ref_freq = (1 - freq) * (1 - freq)
            ref_alt_freq = 1 - ref_freq - alt_freq
        if ref_freq >= alt_freq and ref_freq >= ref_alt_freq:
            return 0
        elif alt_freq >= ref_freq and alt_freq >= ref_alt_freq:
            return 2
        else:
            return 1
    if no_none:
        return 0
    return None


def fill_genotypes(genotypes, snp_maf, snp_list=None):
    """
    Fill NA to most frequent genotype.

    :param genotypes:
    :param snp_maf:
    :param snp_list
    :return:
    """
    to_drop = []
    if not snp_list:
        snp_list = genotypes.columns
    for rs in snp_list:
        if rs in snp_maf.index:
            m = snp_maf.loc[rs]
            gt = get_most_freq_genotype(m, True)
        else:
            to_drop.append(rs)
            continue
        if rs in genotypes.columns:
            c = genotypes[rs]
            c.fillna(gt, inplace=True)
        else:
            genotypes[rs] = gt
    if len(to_drop) > 0:
        genotypes = genotypes.drop(to_drop)
    return genotypes


def drop_snps(genotypes, row_thresh=None, col_thresh=None):
    """
    Drop snp if number of NA is more than thresh.

    :param genotypes:
    :param row_thresh:
    :param col_thresh:
    :return:
    """
    return genotypes.dropna(axis=0, thresh=row_thresh).dropna(axis=1, thresh=col_thresh)


def get_genotypes(filepath, mafpath=None, snp_list=None, **kwargs):
    """
    Get genotype DataFrame.

    :param filepath:
    :param mafpath:
    :param snp_list:
    :param kwargs:
    :return:
    """
    if mafpath is None:
        mafpath = os.path.split(os.path.realpath(__file__))[0] + '/lib/maf.csv'
    maf = load_mafs(mafpath)
    gt = process_genotypes(filepath, snp_maf=maf, snp_list=snp_list, **kwargs)
    if snp_list and 'min_snp' in kwargs:
        minsnp = int(kwargs['min_snp'])
        if len(gt.columns) < minsnp:
            raise Exception('SNP number (%d) is not enough' % len(gt.columns))
    if 'drop_snp' in kwargs and kwargs['drop_snp']:
        rt = kwargs['drop_snp_row_thresh'] if 'drop_snp_row_thresh' in kwargs else None
        ct = kwargs['drop_snp_col_thresh'] if 'drop_snp_col_thresh' in kwargs else None
        gt = drop_snps(gt, row_thresh=rt, col_thresh=ct)
    if 'fill_genotype' in kwargs and kwargs['fill_genotype']:
        gt = fill_genotypes(gt, snp_maf=maf, snp_list=snp_list)
    return gt


def load_data(filepath, gender=True):
    """
    Load CSV data from parse_inputs.py script output.
    x contains genotypes and gender
    y contains height

    :param filepath:
    :param gender
    :return: x, y
    """
    data = pd.read_csv(filepath, index_col=0)
    if gender:
        return split_genotype_height(data)
    else:
        return split_genotype_height_without_gender(data)


def split_genotype_height(data):
    """
    Split DataFrame to features and target.

    :param data:
    :return:
    """
    height = data[['height']]
    gt = data.drop('height', axis=1)
    enc = OneHotEncoder(3)
    x = enc.fit_transform(gt).toarray()
    y = np.array(height['height'])
    return x, y


def split_genotype_height_without_gender(data):
    """
    Split DataFrame to height and genotypes without gender.

    :param data:
    :return:
    """
    height = data[['height']]
    gt = data.drop('height', axis=1).drop('gender', axis=1)
    enc = OneHotEncoder(3)
    x = enc.fit_transform(gt).toarray()
    y = np.array(height['height'])
    return x, y
