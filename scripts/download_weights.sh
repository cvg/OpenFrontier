gdown 11SugqEg3LR2voKdLvq9Xe_zch10ek006
mkdir -p model_weights
mv rgbd_11cls.pth model_weights/
cd model_weights
wget https://raw.githubusercontent.com/bdaiinstitute/vlfm/refs/heads/main/data/pointnav_weights.pth