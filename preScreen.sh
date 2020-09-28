full_root = "../ResearchData/UltraImageUSFullTest/UltraImageCropFull"
tran_root = "../ResearchData/UltraImageUSFullTest/UltraImageCropTransection"
vertical_root = "../ResearchData/UltraImageUSFullTest/UltraImageCropVertical"
python ./1i1oSeg.py --s_data_root $full_root --criterion TheCrossEntropy
python ./1i1oSeg.py --s_data_root $tran_root --criterion TheCrossEntropy
python ./1i1oSeg.py --s_data_root $vertical_root --criterion TheCrossEntropy
python ./1i1oSeg.py --s_data_root $full_root --criterion IOULoss
python ./1i1oSeg.py --s_data_root $tran_root --criterion IOULoss
python ./1i1oSeg.py --s_data_root $vertical_root --criterion IOULoss
python ./1i1oSeg.py --s_data_root $full_root --criterion GDL
python ./1i1oSeg.py --s_data_root $tran_root --criterion GDL
python ./1i1oSeg.py --s_data_root $vertical_root --criterion GDL