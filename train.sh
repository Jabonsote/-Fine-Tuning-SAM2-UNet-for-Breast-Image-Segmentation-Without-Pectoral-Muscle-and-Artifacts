CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "sam2_hiera_large.pt" \
--train_image_path "Dataset/train/images/" \
--train_mask_path "Dataset/train/masks/" \
--save_path "checkpoint" \
--epoch 1000 \
--lr 0.001 \
--batch_size 16 
  
