python main.py --name cityscapes/swiftnet_rn18/reinforce20 --backbone resnet18 \
--segblocks-policy reinforce --segblocks-percent-target 0.2  \
--mode val --resume-best y --viz policy pred
