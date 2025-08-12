脚本形参：
```shell
--model informer --data QianTangRiver2020-2024WorkedFull --attn prob --freq h --device 0
```

命令
```shell
python main_informer.py --model informer --data custom --root_path ./data/ETT/ --data_path sensor_data_20240601.csv --task_type classification --target abnormal_type --enc_in 13 --c_out 6 --num_classes 6 --features M --devices 0
```