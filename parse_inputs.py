import numpy as np
import pandas as pd
import datetime
import argparse
import os
import preprocess


def load_specimen(filepath, min_height=None, min_age=None):
    """

    :param filepath:
    :param min_height:
    :param min_age:
    :return: heights, genders, ages
    :rtype: pd.Series, pd.Series, pd.Series
    """
    with open(filepath, encoding='utf-8') as fh:
        genders = {}
        heights = {}
        ages = {}
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cols = line.split('\t')  # accession name gender height dob age
            acc = cols[0]
            # parse height
            if len(cols) < 4:
                # no height
                continue
            h = float(cols[3])
            if min_height is not None and h < min_height:
                continue
            else:
                heights[acc] = h
            # parse age
            age = int(cols[5]) if len(cols) > 5 else None
            if age is None and len(cols) > 4:
                dob = datetime.datetime.strptime(cols[4], '%Y-%m-%d')
                age = datetime.datetime.today().year - dob.year
            if min_age is not None and (age is None or age < min_age):
                continue
            ages[acc] = age
            # parse gender, skip unknown gender
            gender = parse_gender(cols[2])
            if gender is None:
                continue
            genders[acc] = gender
        return pd.Series(heights), pd.Series(genders), pd.Series(ages, dtype=np.uint8)


def parse_gender(gender):
    """
    Parse gender, 1 for male, 0 for female.
    :param gender:
    :return:
    """
    gender = gender.lower()
    if gender == 'male' or gender == 'm':
        return 1
    elif gender == 'female' or gender == 'f':
        return 0
    else:
        return None


def parse_snp_list(snp_list_type):
    """
    Parse snp list.

    :param snp_list_type:
    :return:
    """
    if snp_list_type == '1996':
        return snp_list_1996
    else:
        return snp_list_119


def init_args(arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--specimen', help='Specimen input file, only for labeled genotypes')
    parser.add_argument('-g', '--genotype', help='Genotype input file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--gender', help='Specify gender for non-label specimen')
    parser.add_argument('-l', '--genotype-label', help='Whether has genotype label', action='store_true', default=False)
    parser.add_argument('--snp-list', help='Choose type of snp list', choices=['119', '1996'], default='119')
    args = parser.parse_args(arguments)
    if not args.genotype or not os.path.exists(args.genotype):
        print('Invalid genotype file')
        return None
    if not args.output:
        print('Invalid output file')
        return None
    return args


def process_one(args=None):
    snp_list = parse_snp_list(args.snp_list)
    genotype = preprocess.get_genotypes(args.genotype, snp_list=snp_list, fill_genotype=True, genotype_label=False,
                                        drop_snp=False)
    gender = parse_gender(args.gender)
    genotype['height'] = 0
    genotype['gender'] = gender
    if snp_list:
        cols = ['height', 'gender'] + snp_list
        genotype = genotype.reindex(columns=cols)
    # save to file
    n_sam, n_fea = genotype.shape
    print("Samples: %d\tFeatures: %d" % (n_sam, n_fea))
    print('Save to file %s' % args.output)
    genotype.to_csv(args.output)


def process_many(args=None):
    snp_list = parse_snp_list(args.snp_list)
    genotype = preprocess.get_genotypes(args.genotype, snp_list=snp_list, fill_genotype=True, genotype_label=True,
                                        drop_snp=True, drop_snp_row_thresh=50, drop_snp_col_thresh=50)
    heights, genders, ages = load_specimen(args.specimen, min_height=100)
    spe = pd.DataFrame({'height': heights, 'gender': genders})
    df = pd.merge(spe, genotype, left_index=True, right_index=True)
    if snp_list:
        cols = ['height', 'gender'] + snp_list
        df = df.reindex(columns=cols)
    # save to file
    n_sam, n_fea = df.shape
    print("Samples: %d\tFeatures: %d" % (n_sam, n_fea))
    print('Save to file %s' % args.output)
    df.to_csv(args.output)


def main(args=None):
    args = init_args(args)
    if args is None:
        return
    if args.genotype_label:
        process_many(args)
    else:
        process_one(args)


snp_list_119 = ['rs11082671', 'rs12612930', 'rs11021504', 'rs3816804', 'rs3751599', 'rs7678436', 'rs3791679',
                'rs9825379', 'rs10513137', 'rs2284746', 'rs3738814', 'rs7513464', 'rs12410416', 'rs1926872',
                'rs1890995', 'rs3769528', 'rs4146922', 'rs7588654', 'rs6753739', 'rs10460436', 'rs13072744',
                'rs7636293', 'rs6763931', 'rs4243400', 'rs6823268', 'rs16895802', 'rs13131350', 'rs2227901',
                'rs3733309', 'rs16848425', 'rs2011962', 'rs2454206', 'rs6845999', 'rs7704138', 'rs7708474', 'rs806794',
                'rs1776897', 'rs1415701', 'rs13273123', 'rs7815909', 'rs2062078', 'rs10858250', 'rs606452',
                'rs11170624', 'rs1971762', 'rs2066808', 'rs2271266', 'rs699371', 'rs7158300', 'rs10519302', 'rs2401171',
                'rs3817428', 'rs2871865', 'rs2573652', 'rs757608', 'rs3785574', 'rs8098316', 'rs4369779', 'rs16950303',
                'rs2145272', 'rs6060369', 'rs2236164', 'rs11205277', 'rs1325598', 'rs4665736', 'rs11694842', 'rs867529',
                'rs611203', 'rs1541777', 'rs10037512', 'rs537930', 'rs4282339', 'rs889014', 'rs7759938', 'rs6570507',
                'rs2510897', 'rs11107116', 'rs7153027', 'rs1659127', 'rs17782313', 'rs6772112', 'rs9818941', 'rs572169',
                'rs526896', 'rs3812163', 'rs12680655', 'rs10512248', 'rs779933', 'rs234886', 'rs1042725', 'rs2093210',
                'rs11648796', 'rs4821083', 'rs2166898', 'rs3823418', 'rs12413361', 'rs17152411', 'rs3781426',
                'rs6030712', 'rs1865760', 'rs11970475', 'rs2251830', 'rs4472734', 'rs3755206', 'rs10448080',
                'rs4733789', 'rs174547', 'rs1938679', 'rs3809128', 'rs7184046', 'rs258324', 'rs2270518', 'rs12459943',
                'rs600130', 'rs41464348', 'rs142036701', 'rs148833559', 'rs148934412', 'rs137852591']

snp_list_1996 = ['rs1000597','rs10010131','rs1001179','rs10033900','rs10036748','rs10037512','rs10052657','rs10069690','rs1007000','rs10076782','rs10086908','rs10088247','rs10090154','rs10114470','rs1011970','rs10128556','rs1015213','rs10157197','rs10160804','rs1016343','rs10165485','rs10168266','rs1018326','rs10192566','rs10229583','rs1024161','rs1026364','rs10273639','rs10306114','rs103294','rs1033667','rs1034589','rs10411210','rs10411619','rs10416218','rs1041981','rs1042026','rs1042522','rs1042602','rs1042636','rs1042713','rs1042714','rs1042725','rs10427255','rs1042858','rs10429035','rs10448080','rs1045642','rs10460436','rs1047286','rs10474352','rs1047781','rs10484560','rs10484761','rs10489177','rs10489202','rs104894275','rs104894276','rs104894277','rs104894280','rs1048943','rs104894398','rs1048990','rs10490924','rs10491327','rs10492336','rs1049346','rs10500565','rs1050450','rs1050565','rs10507522','rs1050828','rs1050829','rs10508372','rs1050976','rs1051168','rs10512248','rs10513137','rs1051375','rs10513801','rs10515552','rs1051730','rs10519302','rs1053004','rs1053051','rs1056892','rs1057910','rs1058164','rs1058319','rs1061235','rs1061622','rs1063192','rs1064395','rs1064651','rs1065852','rs10733310','rs10739221','rs10743108','rs10750097','rs10757272','rs10757274','rs10757278','rs10759243','rs10771399','rs10774214','rs10781182','rs10794532','rs10795668','rs10798036','rs10800746','rs10806425','rs1080985','rs10811661','rs10818854','rs10821944','rs10822013','rs10823607','rs10830962','rs10830963','rs10842262','rs10845606','rs10847697','rs10849432','rs10850219','rs10858250','rs10859871','rs10865331','rs10876864','rs10885122','rs10886471','rs10888501','rs10889353','rs10892258','rs10893872','rs10903122','rs10906115','rs10911021','rs10911390','rs10922109','rs10923931','rs10929302','rs10931936','rs10932125','rs10936599','rs10937273','rs10937405','rs10941679','rs10947262','rs10948347','rs10965235','rs10980705','rs10985349','rs10986105','rs10993994','rs10994415','rs10995245','rs10995251','rs11005497','rs11021504','rs11023787','rs11024102','rs11030104','rs11031731','rs11038167','rs11038172','rs11041683','rs11046205','rs11057830','rs11065756','rs11065783','rs11066105','rs11066280','rs11067763','rs11073058','rs11075995','rs11080055','rs11082671','rs11100479','rs111033196','rs111033204','rs111033220','rs111033305','rs111033307','rs111033313','rs111033318','rs111033401','rs1110701','rs11107116','rs1111875','rs11142387','rs11145951','rs1116357','rs11170624','rs1117707','rs11190870','rs11191548','rs11191580','rs11195128','rs11196172','rs11199914','rs11204213','rs11204971','rs11205277','rs11206226','rs11206510','rs11216126','rs11218343','rs1121980','rs11221332','rs11235604','rs11240572','rs11249433','rs11265260','rs1127354','rs112754928','rs1130534','rs113994095','rs113994097','rs113994098','rs114092250','rs1141718','rs1142','rs1142345','rs1143679','rs11539086','rs1154155','rs11549465','rs11556218','rs11556924','rs11574311','rs1159268','rs11593576','rs11598564','rs11604680','rs11605924','rs1160985','rs11615','rs11634397','rs11640439','rs11648796','rs11649743','rs11668344','rs1166861','rs11671664','rs116855232','rs1169310','rs11694842','rs11697325','rs117026326','rs11708067','rs11712165','rs11717455','rs11722228','rs1172294','rs11726117','rs117385606','rs11746443','rs11748327','rs11752643','rs11755164','rs117601636','rs117633859','rs11787792','rs11805303','rs11806638','rs118092776','rs11833579','rs1183910','rs11842874','rs11851414','rs11875687','rs11884770','rs11887534','rs11889341','rs11900673','rs11902138','rs11903757','rs119103229','rs11911275','rs11920090','rs1192201','rs1192415','rs1192691','rs11931074','rs11933944','rs11966200','rs11970475','rs11987678','rs12000501','rs12010175','rs12019136','rs1202524','rs12050217','rs12068671','rs12085366','rs12097821','rs12101261','rs12118297','rs12139042','rs12142280','rs12190287','rs121907990','rs121907993','rs121907994','rs121908362','rs121913059','rs12202969','rs12205363','rs12219125','rs12229654','rs12229663','rs12248560','rs12272004','rs12290811','rs12296850','rs1229984','rs1234315','rs12357257','rs12379034','rs12410416','rs12413361','rs12413409','rs12413624','rs12422552','rs12424086','rs12425791','rs12447690','rs12449858','rs12459943','rs12461589','rs12463617','rs12471454','rs12478601','rs12484776','rs12493607','rs12494314','rs1250552','rs12524865','rs12525220','rs12526453','rs12529514','rs12529935','rs12537','rs1254319','rs12549902','rs12567209','rs12571751','rs12593813','rs12597579','rs1260326','rs12603526','rs12610495','rs12612930','rs12614','rs12617656','rs12621278','rs12628403','rs12634229','rs1265181','rs12653946','rs12654264','rs1265883','rs12680655','rs12686004','rs12688220','rs12700667','rs12708980','rs12710696','rs12716641','rs12720461','rs12720826','rs12721054','rs12721627','rs12721629','rs12722','rs12740374','rs12762303','rs12762549','rs1278769','rs12807809','rs12821008','rs12821256','rs12822507','rs1286083','rs12901499','rs12904216','rs12913538','rs12913832','rs12917712','rs1292011','rs12928822','rs12934922','rs12945290','rs12946942','rs12970134','rs13003464','rs13020329','rs13041247','rs13042395','rs13065203','rs13068101','rs13072744','rs13096142','rs13098911','rs13117307','rs13119723','rs13131350','rs13132308','rs13132933','rs13143871','rs13146789','rs13151961','rs13153971','rs131654','rs13192471','rs13194053','rs13202464','rs13210693','rs13211507','rs1321311','rs13245690','rs13248513','rs13254738','rs1325598','rs13266634','rs13273123','rs13278062','rs13292136','rs13314993','rs1333042','rs1333048','rs13336428','rs13361707','rs13382811','rs13385731','rs13387042','rs13394619','rs13397','rs13405728','rs13429458','rs1353248','rs1359062','rs1359790','rs1373004','rs137852327','rs137852340','rs137852341','rs137852342','rs137852591','rs137853280','rs137853284','rs137853287','rs1378938','rs1378942','rs1380703','rs138105638','rs1384236','rs138694505','rs138740','rs1393820','rs1397829','rs140647181','rs1412444','rs1413299','rs1414334','rs1415701','rs1417210','rs1418190','rs141853578','rs1419881','rs1420101','rs142036701','rs1426654','rs1430740','rs1432679','rs143383','rs1436953','rs1447295','rs1456988','rs146109287','rs1465618','rs146600671','rs146922325','rs1470579','rs147689733','rs147859257','rs1478693','rs147952620','rs148553336','rs148611340','rs148833559','rs148934412','rs1495965','rs1501908','rs1511412','rs151193009','rs1512268','rs1517114','rs151823','rs1531343','rs1537377','rs1538138','rs1539808','rs1540771','rs1541777','rs1544410','rs1546124','rs1547374','rs1551570','rs1552224','rs1558902','rs1562430','rs1564282','rs156429','rs1564981','rs1566045','rs1569699','rs1569723','rs1571878','rs1572072','rs1617640','rs1626340','rs163171','rs163184','rs1635','rs1642764','rs1656404','rs1659127','rs1663689','rs1667394','rs167770','rs16848425','rs16857609','rs16858082','rs16861329','rs16887244','rs16891982','rs16895802','rs16896923','rs16901979','rs16940186','rs16940212','rs16946160','rs16947','rs16948876','rs1695','rs16950303','rs16957913','rs16969681','rs16973225','rs16976358','rs16992204','rs16998073','rs16999165','rs17006292','rs17008493','rs1701704','rs170183','rs17026156','rs17030613','rs17035378','rs17040773','rs17044858','rs17070145','rs17071124','rs17085007','rs17095830','rs17101923','rs17106184','rs17118','rs17145738','rs17152411','rs17173197','rs17190618','rs17231506','rs17242787','rs17244841','rs17249754','rs17264332','rs1726866','rs1727313','rs17297851','rs17300539','rs17356907','rs17366568','rs17376456','rs17401966','rs17450969','rs174537','rs174547','rs174549','rs174550','rs17457593','rs1746048','rs17465637','rs17513503','rs17563','rs17584499','rs17608059','rs176095','rs17653722','rs17706633','rs17728338','rs17728461','rs17744182','rs17761864','rs1776897','rs17782313','rs17810546','rs17822931','rs17826816','rs17827807','rs17863778','rs17868323','rs17868324','rs1794275','rs1795061','rs1799732','rs1799750','rs1799752','rs1799853','rs1799945','rs1799964','rs1799971','rs1799978','rs1799983','rs1800312','rs1800371','rs1800414','rs1800460','rs1800462','rs1800469','rs1800497','rs1800566','rs1800584','rs1800588','rs1800624','rs1800625','rs1800747','rs1800779','rs1800795','rs1800813','rs1800872','rs1800955','rs1801019','rs1801131','rs1801133','rs1801176','rs1801252','rs1801253','rs1801260','rs1801274','rs1801282','rs1801394','rs1802295','rs1805007','rs1805094','rs1815739','rs1820453','rs182429','rs182549','rs1828522','rs1837253','rs1842896','rs1845900','rs1852469','rs1854853','rs1858037','rs1861411','rs1864152','rs1865760','rs187060802','rs1871054','rs1871859','rs187238','rs1877455','rs1878526','rs1883025','rs1883414','rs1887320','rs1890995','rs1893217','rs1893592','rs1894116','rs1894406','rs1898671','rs1902859','rs1906953','rs191281603','rs1913517','rs192366176','rs1926872','rs1938679','rs1949200','rs1952706','rs1953558','rs195422','rs1954787','rs1964463','rs1971762','rs1975197','rs1980422','rs1982963','rs1990760','rs199475579','rs199475584','rs199475645','rs199475659','rs199475687','rs2000811','rs200455203','rs200519781','rs200735402','rs200990725','rs201038679','rs2011425','rs2011962','rs201377835','rs2014300','rs201503','rs201562855','rs2016520','rs201660407','rs202011870','rs2023532','rs2025804','rs2028299','rs2030519','rs2031920','rs2032582','rs2032583','rs2033562','rs2034650','rs2038823','rs20417','rs2043085','rs2046210','rs2048327','rs204993','rs204995','rs20541','rs2055979','rs2057291','rs2059807','rs2062078','rs2066808','rs2069408','rs2069514','rs2069762','rs2070600','rs2070895','rs2070995','rs2071652','rs2072134','rs2072590','rs2072671','rs2073716','rs2074038','rs2074356','rs2075252','rs2075650','rs2075876','rs2076295','rs2076531','rs2093210','rs2097282','rs2106261','rs2108225','rs2108622','rs2114039','rs2116260','rs2123536','rs212402','rs213210','rs2142731','rs2143340','rs2143571','rs2145272','rs2153271','rs2153713','rs2155219','rs2157719','rs2164983','rs2166898','rs2170527','rs2187668','rs2191566','rs2200733','rs2205960','rs2206734','rs2227901','rs2228001','rs2228479','rs2228570','rs2229109','rs2230345','rs2230500','rs2230806','rs2230926','rs2231142','rs2232228','rs2233424','rs2233580','rs2234693','rs2234962','rs2235371','rs2236007','rs2236164','rs2236313','rs2236907','rs2237717','rs2237892','rs2237895','rs2237897','rs2238149','rs2239612','rs2239815','rs2240017','rs2240419','rs2241201','rs224126','rs2241766','rs2241767','rs2241880','rs2242480','rs2244012','rs2248462','rs2248932','rs225013','rs2251830','rs2252996','rs2253206','rs2253907','rs2255280','rs2259816','rs2267633','rs2267668','rs2268361','rs2269577','rs2270518','rs2270968','rs2271266','rs227163','rs2272046','rs2273017','rs2273650','rs2274223','rs227425','rs2274976','rs2275606','rs2276405','rs2279343','rs2280543','rs2281388','rs2281389','rs2281617','rs2281845','rs2282679','rs2284038','rs2284746','rs2285947','rs228697','rs2289669','rs2290203','rs2291569','rs2293035','rs2294008','rs2294025','rs229527','rs2297235','rs2297595','rs2298585','rs2300478','rs2301888','rs2303790','rs2305795','rs2306283','rs2331291','rs234853','rs234886','rs2349415','rs2363956','rs2367894','rs2387397','rs2389995','rs2393903','rs2395029','rs2395185','rs2401171','rs2412971','rs241453','rs2415317','rs2423279','rs243021','rs2431697','rs2439302','rs2447739','rs2454206','rs2469434','rs2474619','rs2476601','rs2477686','rs2479106','rs2483280','rs2488457','rs249038','rs2510897','rs2517532','rs2517892','rs2523946','rs2531995','rs2535633','rs25487','rs2555364','rs2569190','rs2573652','rs2576037','rs2582532','rs258324','rs2592551','rs2596542','rs261967','rs2634073','rs2647012','rs26722','rs267606617','rs267606619','rs267608289','rs2688419','rs2725338','rs2725544','rs2730260','rs2732547','rs2733052','rs2736098','rs2736100','rs2736340','rs2738048','rs2738058','rs2738446','rs273898676','rs2740488','rs2740574','rs2746394','rs2753325','rs275651','rs276174813','rs276174825','rs276174826','rs276174871','rs276174875','rs276174889','rs276174902','rs276174903','rs2764264','rs2781377','rs2787417','rs2794520','rs2799077','rs28362680','rs28371685','rs28371699','rs28371706','rs28371725','rs28371759','rs28399499','rs28399504','rs2841277','rs2842339','rs2844682','rs2847281','rs2847297','rs2853677','rs2854536','rs2856717','rs2856718','rs2857151','rs2857595','rs2858308','rs2860580','rs28642252','rs2867461','rs2871865','rs28727938','rs28735595','rs2887571','rs28897743','rs28897756','rs2890982','rs28940868','rs2894207','rs28942074','rs2895680','rs2896019','rs2897442','rs290487','rs29232','rs2925979','rs2934498','rs2937573','rs2943641','rs2946504','rs2952768','rs2954029','rs296547','rs296766','rs2969180','rs2981579','rs2995264','rs3019885','rs3021304','rs3025058','rs3057','rs3077','rs3087243','rs3088362','rs3093953','rs3094187','rs3095324','rs310241','rs3102734','rs3110232','rs3117008','rs3117039','rs3117098','rs3118470','rs3118520','rs3125734','rs3126085','rs3129055','rs3129269','rs3129860','rs3129871','rs3129878','rs3129882','rs3129890','rs3129900','rs3129943','rs3130299','rs3130501','rs3130542','rs3131000','rs3131296','rs3132613','rs31344','rs3135363','rs3138141','rs3174808','rs3181374','rs3184504','rs3188055','rs3208181','rs3211371','rs3212986','rs3213094','rs322458','rs324420','rs328','rs335206','rs339331','rs33962847','rs33972313','rs340630','rs340874','rs34330','rs344081','rs34593439','rs34778348','rs34920465','rs35145938','rs35292876','rs35444','rs35597309','rs35599367','rs35652124','rs35668226','rs35742686','rs35744813','rs35837782','rs36208070','rs362540','rs363050','rs364897','rs36600','rs372883','rs3733309','rs3733606','rs3736228','rs3738814','rs3741208','rs3745274','rs3745516','rs3747517','rs3750518','rs3751385','rs3751599','rs3753841','rs3755206','rs3757824','rs3760776','rs3761121','rs3761959','rs3764147','rs3764261','rs3769528','rs377457','rs3781264','rs3781426','rs3781834','rs3781913','rs3782886','rs3782889','rs3783782','rs3785574','rs3786897','rs3791679','rs3801387','rs3802379','rs3802457','rs3802842','rs3803662','rs3803800','rs3803915','rs3805322','rs38055','rs3806156','rs3806624','rs3807989','rs3809128','rs3809263','rs3810291','rs3810622','rs3810936','rs3812163','rs3813929','rs3815148','rs3816804','rs381737','rs3817428','rs3817963','rs381815','rs3823418','rs3824999','rs3825214','rs3825932','rs3827066','rs3852724','rs3864302','rs3873385','rs3890324','rs3892097','rs389261','rs3893464','rs3905706','rs3909184','rs391300','rs3914132','rs3918242','rs3918290','rs3923113','rs396991','rs397507419','rs397507420','rs397507563','rs397507612','rs397507614','rs397507627','rs397507823','rs397507922','rs397508011','rs397508022','rs397508983','rs397508987','rs397508993','rs397509011','rs397509035','rs397509058','rs397509060','rs397509301','rs397514401','rs397514420','rs397514425','rs397516416','rs397516878','rs398122576','rs399593','rs401681','rs402710','rs40457','rs404860','rs4072037','rs4072111','rs4112788','rs412657','rs41268474','rs41281112','rs41291556','rs41293455','rs41293497','rs4133274','rs4140564','rs4141001','rs4141819','rs4142110','rs4143334','rs41464348','rs4146922','rs4148323','rs41483948','rs4149056','rs420259','rs421016','rs4227','rs4233949','rs4242382','rs4243400','rs4244285','rs4245595','rs4248154','rs42490','rs4275659','rs4282339','rs4282438','rs4293393','rs429358','rs429608','rs4301434','rs4313034','rs4338740','rs4343','rs4344','rs4353229','rs4369779','rs4369815','rs4385527','rs439401','rs4394274','rs4402960','rs4410790','rs4420638','rs442309','rs4424296','rs4428898','rs4430796','rs443198','rs4444235','rs4445406','rs4453791','rs4455882','rs4457053','rs4462937','rs4472734','rs4481887','rs4488809','rs4489954','rs4491709','rs4523957','rs4538475','rs4552569','rs455804','rs4565946','rs4574921','rs459193','rs4607103','rs4607517','rs4618210','rs4622329','rs463426','rs4635554','rs4639966','rs4646','rs4646438','rs4654748','rs4654903','rs465498','rs4664308','rs4665736','rs4673993','rs4675374','rs4680','rs4698412','rs4711332','rs4713607','rs4713693','rs4713766','rs4713916','rs4722404','rs4727338','rs4728142','rs4733220','rs4733789','rs4736437','rs4752781','rs4757391','rs4765913','rs4776970','rs4778138','rs4778879','rs4778889','rs4779520','rs4779584','rs478304','rs4783244','rs4784165','rs4784227','rs4785204','rs4789937','rs4790881','rs479336','rs479844','rs4804803','rs4808801','rs4809957','rs4813003','rs4813802','rs4819388','rs4821083','rs4821116','rs4821124','rs4821940','rs4822983','rs4838605','rs4842044','rs4845625','rs4846914','rs4849887','rs4852324','rs4880','rs4884660','rs4885093','rs4894535','rs489693','rs4910907','rs4911414','rs4911442','rs4915077','rs4916219','rs4917014','rs4925386','rs4939827','rs4947296','rs4948496','rs4950','rs4950928','rs4951011','rs4959053','rs4961','rs4973768','rs4977574','rs498136','rs498422','rs4985155','rs4986893','rs4986909','rs4987161','rs4988235','rs4988321','rs498872','rs5009448','rs501120','rs5019252','rs5020946','rs5029939','rs5030359','rs5030655','rs5030656','rs5030849','rs5030859','rs5030862','rs5030865','rs5030867','rs5051','rs505151','rs505802','rs505922','rs5065','rs507666','rs5082','rs510769','rs514315','rs515071','rs515983','rs5182','rs5186','rs519113','rs5215','rs5219','rs526896','rs527616','rs529565','rs53576','rs5370','rs537930','rs5400','rs5443','rs544968','rs547154','rs548234','rs551060742','rs556621','rs55743914','rs55886062','rs55901263','rs55914911','rs55951658','rs55975637','rs560887','rs56219066','rs56252704','rs56276455','rs56337013','rs570618','rs570963','rs572169','rs574367','rs5743708','rs574808','rs5749286','rs5754227','rs5759167','rs5764455','rs5768709','rs579459','rs57968458','rs5817082','rs591166','rs5912838','rs59421388','rs5945326','rs599839','rs600130','rs6006611','rs6010620','rs601338','rs6017317','rs6022786','rs602662','rs6030712','rs603446','rs60369023','rs6046','rs6060369','rs606452','rs6065','rs610037','rs6102059','rs611203','rs614367','rs616488','rs6184','rs61907765','rs61941274','rs61985136','rs62247658','rs62358361','rs625039','rs62508588','rs62514907','rs62514927','rs62514955','rs62516095','rs6265','rs6269','rs6311','rs63750006','rs63750114','rs638893','rs6413504','rs6444895','rs6457580','rs6457617','rs6467136','rs646776','rs647161','rs6472903','rs6478106','rs6478108','rs6498114','rs6503659','rs6511720','rs651821','rs652888','rs6540223','rs6542095','rs6545814','rs654690','rs6547741','rs6565597','rs6570507','rs657075','rs6590330','rs6603859','rs660895','rs662799','rs6677604','rs6687758','rs6703183','rs6705628','rs6706649','rs671','rs6715106','rs67156297','rs6721961','rs67376798','rs6753456','rs6753739','rs6754640','rs6763931','rs67666821','rs6772112','rs6773854','rs6774494','rs67784355','rs6780220','rs6780569','rs67839313','rs6793295','rs679620','rs679711','rs6804441','rs6813195','rs6815464','rs6822844','rs6823268','rs6828523','rs6832151','rs6840978','rs6842241','rs6843082','rs6845999','rs6856616','rs6896064','rs6900578','rs6902119','rs6903130','rs6903608','rs6903956','rs6909253','rs6925433','rs6957263','rs6983267','rs699371','rs7007032','rs7016880','rs702634','rs7034200','rs704010','rs704017','rs7041847','rs7044529','rs705379','rs705702','rs7071206','rs7075349','rs7084402','rs7084921','rs7086803','rs708723','rs7087591','rs7097397','rs7104791','rs7105934','rs7107217','rs7108738','rs71325459','rs713598','rs7136259','rs71507014','rs71508903','rs7153027','rs7158300','rs716274','rs7172432','rs7178572','rs7184046','rs718712','rs7190997','rs7194886','rs7197475','rs7204132','rs7206735','rs721048','rs7216064','rs7216389','rs7217932','rs7221059','rs7221571','rs722258','rs7229639','rs72474224','rs72549346','rs72549351','rs72549354','rs72552267','rs72552713','rs72552799','rs72554664','rs72554665','rs72558186','rs72558187','rs725613','rs726288','rs727428','rs72802342','rs7291467','rs72921001','rs72933970','rs7296288','rs730012','rs73036519','rs73058713','rs731236','rs7316898','rs73230612','rs7329174','rs7350481','rs735853','rs73718779','rs738491','rs739496','rs7403531','rs7412','rs74225573','rs74315319','rs743572','rs7447927','rs7453920','rs746427774','rs74735889','rs747650','rs7479949','rs750019452','rs7501331','rs750188782','rs7501939','rs7504990','rs7513464','rs751728','rs752092','rs7521902','rs7528684','rs7531806','rs753955','rs7550047','rs7553711','rs7566605','rs757110','rs7574865','rs757608','rs7578597','rs75796365','rs75804782','rs758115611','rs7584262','rs75842709','rs7588654','rs7593730','rs7594220','rs760426','rs7606754','rs7613051','rs7615568','rs7616215','rs7617456','rs7620503','rs762551','rs76296470','rs763035','rs7630373','rs7631358','rs76337351','rs7634389','rs7636293','rs7639618','rs76418789','rs7642134','rs7656416','rs7660345','rs76687508','rs7671167','rs7677751','rs7678436','rs7679673','rs7682241','rs768245266','rs76830965','rs7686660','rs769258','rs769499327','rs7698250','rs7701094','rs7701890','rs770263702','rs770318608','rs7704138','rs7708474','rs7709212','rs7712641','rs773528125','rs7739264','rs7741604','rs7744813','rs7745098','rs7748270','rs77502020','rs7752903','rs7754840','rs7755545','rs7758229','rs7759482','rs7759938','rs776280797','rs7765379','rs776746','rs7775228','rs7777484','rs7779029','rs7787362','rs778749563','rs7791481','rs779323689','rs779933','rs780094','rs7803454','rs7812088','rs7812879','rs7815909','rs782608284','rs7829127','rs7837688','rs7837791','rs78378222','rs7850258','rs7851693','rs786204474','rs786204581','rs786204739','rs7865618','rs7875560','rs7900194','rs7901695','rs7903146','rs7913069','rs791595','rs7921977','rs7924176','rs7927894','rs7944584','rs7953528','rs7961152','rs7961581','rs79738337','rs7975232','rs7975788','rs79758729','rs798766','rs79931499','rs7997012','rs79976124','rs800292','rs801112','rs80142782','rs802734','rs8030379','rs8032158','rs80338684','rs80338717','rs80338720','rs80338725','rs80338939','rs80338943','rs8034191','rs80356484','rs80356925','rs80356932','rs80357069','rs80357071','rs80357167','rs80357251','rs80357303','rs80357305','rs80357468','rs80357508','rs80357516','rs80357567','rs80357579','rs80357602','rs80357618','rs80357678','rs80357702','rs80357721','rs80357772','rs80357783','rs80357786','rs80357808','rs80357814','rs80357842','rs80357853','rs80357868','rs80357885','rs80357887','rs80357908','rs80357973','rs80357996','rs80358004','rs80358041','rs80358042','rs80358043','rs80358053','rs80358126','rs80358131','rs80358419','rs80358474','rs80358515','rs80358557','rs80358785','rs80358815','rs80358981','rs80359013','rs80359099','rs80359305','rs80359309','rs80359338','rs80359351','rs80359372','rs80359376','rs80359391','rs80359403','rs80359463','rs80359475','rs80359480','rs80359490','rs80359527','rs80359531','rs80359544','rs80359597','rs80359604','rs80359605','rs80359659','rs80359671','rs80359675','rs80359747','rs80359770','rs8042680','rs806215','rs8067378','rs806794','rs8083786','rs8085365','rs8098316','rs81002873','rs81002874','rs8135665','rs8141797','rs8176746','rs817826','rs8192678','rs8192709','rs820430','rs823128','rs823156','rs831571','rs835784','rs855791','rs855867','rs861020','rs864745','rs867529','rs872071','rs873549','rs877819','rs878860','rs880315','rs885479','rs888208','rs888789','rs889014','rs889140','rs889312','rs894278','rs896854','rs897057','rs897200','rs909116','rs910049','rs915404','rs9260734','rs9263739','rs9266406','rs9268402','rs9268839','rs9271100','rs9271366','rs9271588','rs9272105','rs9272346','rs9275224','rs9275319','rs9275572','rs9275596','rs9275698','rs9277535','rs9277952','rs928674','rs9288318','rs9296095','rs9296249','rs9299','rs9302752','rs9304742','rs9307551','rs930847','rs9308731','rs9309245','rs9314614','rs9316871','rs9322335','rs9332131','rs9344996','rs9349379','rs9350504','rs9355610','rs9356744','rs9357155','rs9357271','rs9368677','rs9387478','rs9394159','rs940187','rs9405098','rs941764','rs9420907','rs943080','rs944289','rs9466056','rs9468925','rs9469099','rs9470794','rs9473924','rs9479482','rs9485372','rs949441','rs9500927','rs9502570','rs9505118','rs9510787','rs9510902','rs951896','rs9533095','rs9533634','rs9561778','rs9564692','rs9568867','rs9600079','rs96067','rs961253','rs9632884','rs9633553','rs963468','rs964184','rs965513','rs966423','rs9693444','rs972283','rs974484','rs974819','rs9792269','rs9808753','rs9810888','rs9815073','rs9818941','rs9825379','rs9828933','rs9834970','rs9841504','rs9851967','rs9854612','rs9870933','rs987525','rs987870','rs9880772','rs9883177','rs9888739','rs9904341','rs9907432','rs9921222','rs9922235','rs9923231','rs9925231','rs9929218','rs9937','rs9938149','rs9939609','rs995322','rs9960767','rs999556']


if __name__ == '__main__':
    main()
