full_root="../ResearchData/UltraImageUSFullTest/UltraImageCropFull"
tran_root="../ResearchData/UltraImageUSFullTest/UltraImageCropTransection"
vertical_root="../ResearchData/UltraImageUSFullTest/UltraImageCropVertical"
python ./1i1oSeg.py --s_data_root $full_root --criterion TheCrossEntropy --logdir F_CE
python ./1i1oSeg.py --s_data_root $tran_root --criterion TheCrossEntropy --logdir F_CE
python ./1i1oSeg.py --s_data_root $vertical_root --criterion TheCrossEntropy --logdir F_CE
python ./1i1oSeg.py --s_data_root $full_root --criterion IOULoss --logdir F_IOU
python ./1i1oSeg.py --s_data_root $tran_root --criterion IOULoss --logdir T_IOU
python ./1i1oSeg.py --s_data_root $vertical_root --criterion IOULoss --logdir V_IOU
python ./1i1oSeg.py --s_data_root $full_root --criterion GDL --logdir F_GDL
python ./1i1oSeg.py --s_data_root $tran_root --criterion GDL --logdir T_GDL
python ./1i1oSeg.py --s_data_root $vertical_root --criterion GDL --logdir V_GDL