import mpathic as mpa

# Load dataset and model dataframes
dataset_df = mpa.io.load_dataset('sort_seq_data.txt')
model_df = mpa.io.load_model('crp_model.txt')

# learn models example
learned_model = mpa.LearnModel(df=dataset_df)
learned_model.output_df.head()

# evaluate models example
eval_model = mpa.EvaluateModel(dataset_df = dataset_df, model_df = model_df)
eval_model.out_df.head()

# scan models example
# get contigs, provided with mpathic
fastafile = "./mpathic/examples/genome_ecoli_1000lines.fa"
contig = mpa.io.load_contigs_from_fasta(fastafile, model_df)

scanned_model = mpa.ScanModel(model_df = model_df, contigs_list = contigs_list)
scanned_model.sitelist_df.head()

# predictive info example
predictive_info = mpa.PredictiveInfo(data_df = dataset_df, model_df = model_df,start=52)