## Train & evaluate

```
python gnn.py --label_rate 0.01
python gnn_da.py --label_rate 0.01 --gamma 0.01
python gnn_specreg.py --label_rate 0.01 --gamma 0.01 --gamma_ss 0.1 --thres_ss 1 --gamma_mfr 0 --thres_mfr 0
python gnn_specreg.py --label_rate 0.01 --gamma 0.01 --gamma_ss 0 --thres_ss 0 --gamma_mfr 0.1 --thres_mfr 1

python gnn.py --label_rate 0.1
python gnn_da.py --label_rate 0.1 --gamma 0.01
python gnn_specreg.py --label_rate 0.1 --gamma 0.01 --gamma_ss 0.1 --thres_ss 1 --gamma_mfr 0 --thres_mfr 0
python gnn_specreg.py --label_rate 0.1 --gamma 0.01 --gamma_ss 0 --thres_ss 0 --gamma_mfr 0.1 --thres_mfr 1

python gnn.py --label_rate 1
python gnn_da.py --label_rate 1 --gamma 0.01
python gnn_specreg.py --label_rate 1 --gamma 0.01 --gamma_ss 0.1 --thres_ss 1 --gamma_mfr 0 --thres_mfr 0
python gnn_specreg.py --label_rate 1 --gamma 0.01 --gamma_ss 0 --thres_ss 0 --gamma_mfr 0.1 --thres_mfr 1
```
