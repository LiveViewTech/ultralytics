import os
import argparse as ap 
from ultralytics import YOLO
import glob  


if __name__ == "__main__":
    ap = ap.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="dataset.yaml", help="path to the labeled dataset. default = dataset.yaml")
    ap.add_argument("-b", "--batch_size", type=int, default=256, help="batch size for training. default = auto-batch")
    ap.add_argument("-e", "--epochs", type=int, default=300, help="number of epochs to train for. default = 300")
    #ap.add_argument("-f", "--freeze-layers", type=int, default=0, help="number of layers to freeze. default = 0")
    ap.add_argument("-m", "--model", default="yolov8s.yaml", help="model to use for training. default = yolov8s")
    ap.add_argument("-p", "--pretrained", default="yolov8s.pt", help="model to use for training. default = yolov8s")
    ap.add_argument("-dv", "--device", nargs='+', default=0, help="devices to use for training")
    ap.add_argument("-s", "--img_size", type=int, default=640, help="image size")
    ap.add_argument("-sp", "--split", default="test", help="validate on train, val/valid or test split")

    ap.add_argument("-j", "--job", default="train", help="train, validate or test")
    ap.add_argument("-t", "--test_image_folder", default="./test_images", help="test image folder")
    ap.add_argument("--disable_wandb", action='store_true')
    ap.add_argument("--iou", type=float, default = 0.5, help="iou threshold in validation or prediction") 
    ap.add_argument("--conf",type=float, default = 0.1, help="confidence threshold for validation or prediction") 

    args = ap.parse_args()
    print('args = ', args) 

    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.job == 'validate' or args.job == 'test':
        model = YOLO(args.pretrained) 
    else:      
        model = YOLO(args.pretrained) 
        #mddeo = model.load(args.pretrained)   
     
        
    # Train the model
    if args.job == 'train': 
        results = model.train(data=args.dataset, device=args.device, batch=args.batch_size, epochs=args.epochs, imgsz=args.img_size) 

    elif args.job == 'validate':  #dataset with GT  
        metrics = model.val(data=args.dataset, device=args.device, batch=args.batch_size, split = args.split, imgsz=args.img_size, conf=args.conf, iou=args.iou) 
        print(metrics.box.map)  # mAP50-95
        print(metrics.box.map50)  # mAP50
        #print(metrics.box.map75)  # mAP75
        print(metrics.box.maps)  # list of mAP50-95 for each category

    elif args.job == 'test':  #dataset w/o GT
        #only support images directly in a folder 
        img_fns = os.listdir(args.test_image_folder)

        for img in img_fns: 
            img_path = os.path.join(args.test_image_folder,img)
            results = model.predict(img_path, device=args.device, save=True, imgsz=args.img_size, conf=args.conf, iou=args.iou) 
            for result in results: 
                boxes = result.boxes  # Boxes object for bounding box outputs
                probs = result.probs  # Probs object for classification outputs

