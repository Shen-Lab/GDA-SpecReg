
```
mkdir ./data ; cd ./data
wget https://stringdb-static.org/download/protein.links.full.v11.5/9606.protein.links.full.v11.5.txt.gz
# and then unzip, same for below
wget https://stringdb-static.org/download/protein.sequences.v11.5/9606.protein.sequences.v11.5.fa.gz
wget https://stringdb-static.org/download/protein.links.full.v11.5/10090.protein.links.full.v11.5.txt.gz
wget https://stringdb-static.org/download/protein.sequences.v11.5/10090.protein.sequences.v11.5.fa.gz
wget https://stringdb-static.org/download/protein.links.full.v11.5/4932.protein.links.full.v11.5.txt.gz
wget https://stringdb-static.org/download/protein.sequences.v11.5/4932.protein.sequences.v11.5.fa.gz
wget https://stringdb-static.org/download/protein.links.full.v11.5/6239.protein.links.full.v11.5.txt.gz
wget https://stringdb-static.org/download/protein.sequences.v11.5/6239.protein.sequences.v11.5.fa.gz
wget https://stringdb-static.org/download/protein.links.full.v11.5/7227.protein.links.full.v11.5.txt.gz
wget https://stringdb-static.org/download/protein.sequences.v11.5/7227.protein.sequences.v11.5.fa.gz
wget https://stringdb-static.org/download/protein.links.full.v11.5/7955.protein.links.full.v11.5.txt.gz
wget https://stringdb-static.org/download/protein.sequences.v11.5/7955.protein.sequences.v11.5.fa.gz
cd ..
```

```
cd ./unsupervised

python main_seq_encoder.py --tgt_species yeast
python main_graph_encoder.py --tgt_species yeast
python main_adv_training.py --tgt_species yeast --adv_train_gamma 0.000001
python main_spectral_regularization.py --tgt_species yeast --adv_train_gamma 0.000001 --spectral_reg_smooth_gamma 1000 --spectral_reg_lowpass_gamma -1 --spectral_reg_gamma2 0.01
python main_spectral_regularization.py --tgt_species yeast --adv_train_gamma 0.000001 --spectral_reg_smooth_gamma -1 --spectral_reg_lowpass_gamma 1000 --spectral_reg_gamma2 0.01

python main_seq_encoder.py --tgt_species fruit_fly
python main_graph_encoder.py --tgt_species fruit_fly
python main_adv_training.py --tgt_species fruit_fly --adv_train_gamma 0.001
python main_spectral_regularization.py --tgt_species fruit_fly --adv_train_gamma 0.001 --spectral_reg_smooth_gamma 0.01 --spectral_reg_lowpass_gamma -1 --spectral_reg_gamma2 0.001
python main_spectral_regularization.py --tgt_species fruit_fly --adv_train_gamma 0.001 --spectral_reg_smooth_gamma -1 --spectral_reg_lowpass_gamma 0.01 --spectral_reg_gamma2 0.001

python main_seq_encoder.py --tgt_species zebrafish
python main_graph_encoder.py --tgt_species zebrafish
python main_adv_training.py --tgt_species zebrafish --adv_train_gamma 0.001
python main_spectral_regularization.py --tgt_species zebrafish --adv_train_gamma 0.001 --spectral_reg_smooth_gamma 10 --spectral_reg_lowpass_gamma -1 --spectral_reg_gamma2 0.001
python main_spectral_regularization.py --tgt_species zebrafish --adv_train_gamma 0.001 --spectral_reg_smooth_gamma -1 --spectral_reg_lowpass_gamma 10 --spectral_reg_gamma2 0.001

python main_seq_encoder.py --tgt_species mouse
python main_graph_encoder.py --tgt_species mouse
python main_adv_training.py --tgt_species mouse --adv_train_gamma 0.000001
python main_spectral_regularization.py --tgt_species mouse --adv_train_gamma 0.0001 --spectral_reg_smooth_gamma 0.01 --spectral_reg_lowpass_gamma -1 --spectral_reg_gamma2 0.01
python main_spectral_regularization.py --tgt_species mouse --adv_train_gamma 0.0001 --spectral_reg_smooth_gamma -1 --spectral_reg_lowpass_gamma 0.01 --spectral_reg_gamma2 0.01
```

```
cd ./semisupervised

python main_seq_encoder.py --tgt_species yeast
python main_graph_encoder.py --tgt_species yeast
python main_adv_training.py --tgt_species yeast --adv_train_gamma 0.000001
python main_spectral_regularization.py --tgt_species yeast --adv_train_gamma 0.000001 --spectral_reg_smooth_gamma 1 --spectral_reg_lowpass_gamma -1 --spectral_reg_gamma2 0.01
python main_spectral_regularization.py --tgt_species yeast --adv_train_gamma 0.000001 --spectral_reg_smooth_gamma -1 --spectral_reg_lowpass_gamma 100 --spectral_reg_gamma2 1

python main_seq_encoder.py --tgt_species fruit_fly
python main_graph_encoder.py --tgt_species fruit_fly
python main_adv_training.py --tgt_species fruit_fly --adv_train_gamma 0.001
python main_spectral_regularization.py --tgt_species fruit_fly --adv_train_gamma 0.001 --spectral_reg_smooth_gamma 0.1 --spectral_reg_lowpass_gamma -1 --spectral_reg_gamma2 0.1
python main_spectral_regularization.py --tgt_species fruit_fly --adv_train_gamma 0.001 --spectral_reg_smooth_gamma -1 --spectral_reg_lowpass_gamma 1 --spectral_reg_gamma2 1

python main_seq_encoder.py --tgt_species zebrafish
python main_graph_encoder.py --tgt_species zebrafish
python main_adv_training.py --tgt_species zebrafish --adv_train_gamma 0.001
python main_spectral_regularization.py --tgt_species zebrafish --adv_train_gamma 0.001 --spectral_reg_smooth_gamma 1 --spectral_reg_lowpass_gamma -1 --spectral_reg_gamma2 0.01
python main_spectral_regularization.py --tgt_species zebrafish --adv_train_gamma 0.001 --spectral_reg_smooth_gamma -1 --spectral_reg_lowpass_gamma 0.1 --spectral_reg_gamma2 0.1

python main_seq_encoder.py --tgt_species mouse
python main_graph_encoder.py --tgt_species mouse
python main_adv_training.py --tgt_species mouse --adv_train_gamma 0.000001
python main_spectral_regularization.py --tgt_species mouse --adv_train_gamma 0.0001 --spectral_reg_smooth_gamma 0.1 --spectral_reg_lowpass_gamma -1 --spectral_reg_gamma2 0.1
python main_spectral_regularization.py --tgt_species mouse --adv_train_gamma 0.0001 --spectral_reg_smooth_gamma -1 --spectral_reg_lowpass_gamma 1 --spectral_reg_gamma2 0.1
```
