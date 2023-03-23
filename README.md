# Edge Deployable Online Domain Adaptation for Underwater Object Detection

## Results and Models

All evaluated on ReefScan$^{\operatorname{pos}}_{\operatorname{test}}$.

|    Technique      |  Adaptation Set  | F2 0.3:0.8  | Download |
| ----------------  | ---------------- | ----------  | -------- |
| YOLOV5 Source     | NA               | $32.54\%$         | ...      | 
| YOLOV5 Relighting | Source and ReefScan$^{\operatorname{pos}}_{\operatorname{train}}$                | $39.99\%$        | [Lightnet](https://drive.google.com/file/d/1FePGYyDSXGzRzV8ndqtsBwDmb6865lzy/view?usp=share_link)      | 
| YOLOV5 Combined | ReefScan$^{\operatorname{pos}}_{\operatorname{train}}$               | $54.8\%$      | [ckpt](https://drive.google.com/file/d/1DRtpsCGYRvAR3E2Et-Ajq1es5AEb36Gh/view?usp=share_link)      | 
| YOLOV5 Combined | ReefScan$^{\operatorname{sub}}_{\operatorname{train}}$               | $52.3\%$         | [ckpt](https://drive.google.com/file/d/1StazCtDdw9RCYyh1ZeOTNeFNFw148FXw/view?usp=share_link)     | 


Darknet is [available.](https://drive.google.com/file/d/1asXelLKrtTU7paLfy_Rsn8kknSXKutZ7/view?usp=share_link)

## Requirements
* pytorch
* mmyolo

## Datasets
**Kaggle**: Kaggle COTS dataset

**AIMS**: AIMS COTS dataset

### Splits

[Google Drive.](https://drive.google.com/drive/folders/11sTpseVmP23_Rw9Yc8YRi18qn6tv8CHu?usp=share_link)

## Testing
AIMS
```
 python evaluate.py --dataset aims_sep
```
Kaggle
```
python evaluate.py --dataset kaggle
```
## Training

Kaggle -> AIMS
```
 python train.py --run-name aims-adaptation-yolov5 --model yolov5 --batch-size 2 --teacher-score-thresh 0.5
 python train.py --run-name aims-adaptation-yolov8 --model yolov8 --batch-size 1 --teacher-score-thresh 0.5
```

Kaggle -> AIMS multiday
```
 python train_multiday.py --run-name aims-adaptation-yolov5 --model yolov5 --batch-size 2 --teacher-score-thresh 0.7
```

Relighting AIMS -> Kaggle
```
 python train_lightnet.py --run-name aims-relighting --batch-size 2
```

## Acknowledgments
The code is based on [DANNet](https://github.com/W-zx-Y/DANNet), [DINO](https://github.com/facebookresearch/dino) and mmyolo.


### Contact
* Djamahl Etchegaray (uqdetche@uq.edu.au)
