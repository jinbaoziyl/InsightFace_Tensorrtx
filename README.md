# InsightFace_Tensorrtx
InsightFace_Tensorrtx aims to implement InsightFace_Pytorch with tensorrt network definition APIs.
The pytorch implementation is [TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)

The two input images used in this project are joey0.ppm and joey1.ppm, download them from [Google Drive.](https://drive.google.com/drive/folders/1ctqpkRCRKyBZRCNwo9Uq4eUoMRLtFq1e).
## Run

```
1. generate arcface-r50.wts from pytorch implementation https://github.com/TreB1eN/InsightFace_Pytorch

git clone https://github.com/TreB1eN/InsightFace_Pytorch
// copy gen_wts.py into repo, then 
cd InsightFace_Pytorch
python genwts.py
// a file 'arcface-r50.wts' will be generated.

2. put arcface-r50.wts into InsightFace_Tensorrtx/data, build and run
git clone https://github.com/jinbaoziyl/InsightFace_Tensorrtx
cd InsightFace_Tensorrtx/

mkdir build
cd build
// put arcface-r50.wts here
cmake ..
make

sudo ./arcface-r50 -s  // build and serialize model to file i.e. 'arcface-r50.engine'
sudo ./arcface-r50 -d  // deserialize model file and run inference.

3. check two images similality scores.
```