import os
import argparse as ap 
from ultralytics import YOLO
import glob  
import pickle 

import cv2 

if __name__ == "__main__":
    ap = ap.ArgumentParser()
    ap.add_argument("-p", "--pretrained", default="best.pt", required=True, help="model to use for training. default = yolov8s")
    ap.add_argument("-dv", "--device", nargs='+', default=0, help="devices to use for training")
    ap.add_argument("--iou", type=float, default = 0.5, help="iou threshold in validation or prediction") 
    ap.add_argument("--conf",type=float, default = 0.5, help="confidence threshold for validation or prediction") 
    ap.add_argument("-v", "--video_file", default=None, help="test video file")
    ap.add_argument("--video_folder", default=None, help="test video folder")
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--save_track_video', action='store_true')  
    ap.add_argument("--save_dir", default="./results", help="directory to save results")
    ap.add_argument("--classes", nargs='+', default= None, type=int, help="classes included in detection and tracking")

    args = ap.parse_args()
    print('args = ', args) 

    if args.video_file is None and args.video_folder is None:
        print("One of video_file and video_folder must be not None") 
        exit(0)  


    model = YOLO(args.pretrained)

    if len(args.save_dir)>0:
        save_dir = args.save_dir
    else:
        save_dir = "./results" 

    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    video_files = os.listdir(args.video_folder) 

    if args.debug:
        video_files = [args.video_file] 

    for v in video_files: 

        result_file = f'{save_dir}/{v}_result.pkl' 
        print(result_file) 
        #input('test') 

        if not args.debug and os.path.exists(result_file) :
            print(f'{result_file} already exists') 
            #pass 
            continue 

        video_file = os.path.join(args.video_folder, v)   
        vid = cv2.VideoCapture(video_file)
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        vid.release()
        if height==width:
            print("Skip fisheye video: ", v) 
            continue 

        if args.classes is None:
            results = model.track(video_file, save=args.debug or args.save_track_video, show=False, tracker="bytetrack.yaml",conf=args.conf,iou=args.iou)  # with ByteTrack
        else:
            results = model.track(video_file, save=args.debug or args.save_track_video, show=False, tracker="bytetrack.yaml", classes=args.classes, conf=args.conf,iou=args.iou)  # with ByteTrack

        
        results_to_save = {}  

        for i, r in enumerate(results): 
            if args.debug:
                print(i) 
                print(r.boxes)   
            classes  = r.boxes.cls.cpu().numpy() 
            confs = r.boxes.conf.cpu().numpy()
            boxes = r.boxes.xywh.cpu().numpy() 
            track_ids = None 
            try:
                track_ids = r.boxes.id.int().cpu().tolist()
            except:
                pass 

            if args.debug: 
                print("classes = ", classes) 
                print("confs = ", confs) 
                print("boxes = ", boxes) 
                print("ids = ", track_ids) 
                        
            results_to_save[i] = [classes,confs,boxes,track_ids]
             
        # Open a file and use dump() 
        with open(result_file, 'wb') as file: 
            # A new file will be created 
            pickle.dump(results_to_save, file)  

        if args.debug: 
            # Read it back and verify it  
            with open(result_file , 'rb') as file: 
                # A new file will be created 
                data = pickle.load(file)  
                print(data) 
        
         
 


        




