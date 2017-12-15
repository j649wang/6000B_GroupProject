python test.py --dataroot ./datasets/breast --name breast_A2B --model cycle_gan --loadSize 32 --fineSize 32 --ngf 32 --ndf 32 --gpu_ids 0 --results_dir baseline_ab
python train.py --dataroot ./datasets/breast/ --name breast --model cycle_gan --pool_size 20 --display_freq 10 --display_id 0 --loadSize 32 --fineSize 32 --gpu_ids 0 --lr 0.001 --nThread 50
