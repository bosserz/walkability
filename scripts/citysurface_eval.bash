python -m val \
    --dataset citysurfaces \
    --bs_val 1 \
    --eval test \
    --eval_folder ../data/raw_images \
    --snapshot ./weights/block_c_10classes.pth \
    --arch ocrnet.HRNet_Mscale \
    --trunk hrnetv2 \
    --result_dir ../data/segmentation_masks_citysurfaces