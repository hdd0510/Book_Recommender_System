cd pre_train/sasrec
python main.py --device=cuda --dataset Books_5
cd ../../
python main.py --pretrain_stage1 --rec_pre_trained_data Movies_and_TV --gpu_num 0
