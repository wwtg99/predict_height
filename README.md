Predict Height
==============

Predict height by gender and genotypes.

# Usage

### Prepare data

#### Training datasets

Specimen file as below: 5 columns (label, name, gender, height, date of birth, age)
```
GW7S0140C01	张三	male	176	1954-03-05	
GW7S0140C02	李四	female	170	1992-12-28	26
GW7S0140C03	王五	male	175	1957-10-01	
```

Genotype file as below: 4 columns (label, rs, allele1, allele2)
```
GW7S0140C01	rs12688220	T	T
GW7S0140C01	rs5912838	A	C
GW7S0140C01	rs137852591	C	C
GW7S0140C01	rs267606617	A	A
GW7S0140C01	rs267606619	C	C
```

Parse files
```
python parse_inputs.py -s <specimen_file> -g <genotype_file> -o <output_file> -l
```

#### Predict datasets

Genotype file as below: 3 columns (rs, allele1, allele2)
```
rs12688220	T	T
rs5912838	A	C
rs137852591	C	C
rs267606617	A	A
rs267606619	C	C
```

Parse file
```
python parse_inputs.py -g <genotype_file> -o <output_file> --gender=<gender>
```

### Train and save models

Use train datasets from parsed train datasets above, will train models and save model to models(default) directory.
```
python train.py <tain_datesets>
```

### Predict

Use target datasets from parsed predict datasets above, will output predicted height by model.
```
python predict.py <target_datasets>
```
