# Train & evaluate

```
python UDAGCN_demo.py --source acm --target dblp
python main_adv.py --source acm --target dblp
python main_specReg.py --source acm --target dblp --gamma_adv 0.2 --thr_smooth 0.01 --gamma_smooth 0.01
python main_specReg.py --source acm --target dblp --gamma_adv 0.2 --thr_mfr 0.75 --gamma_smooth 0.1

python UDAGCN_demo.py --source dblp --target acm
python main_adv.py --source dblp --target acm
python main_specReg.py --source dblp --target acm --gamma_adv 0.01 --thr_smooth 0.01 --gamma_smooth 0.1
python main_specReg.py --source dblp --target acm --gamma_adv 0.01 --thr_mfr 0.01 --gamma_smooth 0.1
```

## Acknowledgements
The experiment is implemented based on https://github.com/GRAND-Lab/UDAGCN.
