python train.py --dataroot ./datasets/finetune_CVC-300 --name cvc300.1 --pretrained_name cut_all_1200r_20000s --lr 0.0001
python train.py --dataroot ./datasets/finetune_CVC-ClinicDB --name clinicdb.1 --pretrained_name cut_all_1200r_20000s --lr 0.0001
python train.py --dataroot ./datasets/finetune_CVC-ColonDB --name colondb.1 --pretrained_name cut_all_1200r_20000s --lr 0.0001
python train.py --dataroot ./datasets/finetune_ETIS-LaribPolypDB --name etis.1 --pretrained_name cut_all_1200r_20000s --lr 0.0001
python train.py --dataroot ./datasets/finetune_Kvasir --name kvasir.1 --pretrained_name cut_all_1200r_20000s --lr 0.0001