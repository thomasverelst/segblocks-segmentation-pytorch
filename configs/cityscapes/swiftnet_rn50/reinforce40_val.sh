python main.py --name cityscapes/swiftnet_rn50/reinforce40 --backbone resnet50 \
--segblocks-policy reinforce --segblocks-percent-target 0.4  \
--mode val --resume-best y
