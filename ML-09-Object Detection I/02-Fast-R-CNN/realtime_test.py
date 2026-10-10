
from pathlib import Path
import argparse
import numpy as np
import cv2
import tensorflow as tf
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "model.keras"
NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
IMG_SIZE = 64

def preprocess(crop):
    return cv2.cvtColor(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def load_crops(split, limit):
    images_dir = DATASET / split / 'images'
    labels_dir = DATASET / split / 'labels'
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f'Missing dataset folders: {images_dir} or {labels_dir}')
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if limit > 0: files = files[:limit]
    xs, ys = [], []
    for path in files:
        label = labels_dir / (path.stem + '.txt')
        if not label.exists(): continue
        im = cv2.imread(str(path))
        if im is None: continue
        h,w = im.shape[:2]
        for line in label.read_text(encoding='utf-8').splitlines():
            parts=line.split()
            if len(parts)!=5: continue
            try:
                cls=int(parts[0]); xc,yc,bw,bh=map(float,parts[1:])
            except ValueError: continue
            if not 0 <= cls < len(NAMES): continue
            x1=max(0,int((xc-bw/2)*w)); y1=max(0,int((yc-bh/2)*h))
            x2=min(w,int((xc+bw/2)*w)); y2=min(h,int((yc+bh/2)*h))
            if x2<=x1 or y2<=y1: continue
            xs.append(preprocess(im[y1:y2,x1:x2])); ys.append(cls)
    if not xs: raise RuntimeError(f'No labeled crops in {images_dir}. Check image names and YOLO labels.')
    return np.asarray(xs,np.float32),np.asarray(ys,np.int32)

def make_model():
    inputs=tf.keras.Input((IMG_SIZE,IMG_SIZE,3))
    x=tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(inputs)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(128,3,padding='same',activation='relu')(x)
    # Fixed-size ROI feature pooling on each pre-cropped region (educational approximation).
    x=tf.keras.layers.MaxPooling2D(pool_size=4)(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(128,activation='relu')(x)
    outputs=tf.keras.layers.Dense(len(NAMES),activation='softmax')(x)
    model=tf.keras.Model(inputs,outputs,name='Fast_RCNN_educational_region_classifier')
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def propose_regions(frame, max_regions):
    h,w=frame.shape[:2]
    boxes=[]
    for scale in (.2,.35,.5):
        bw=max(16,int(w*scale)); bh=max(16,int(h*scale))
        for cy in np.linspace(bh//2,max(bh//2,h-bh//2),4):
            for cx in np.linspace(bw//2,max(bw//2,w-bw//2),5):
                x1=max(0,int(cx-bw/2)); y1=max(0,int(cy-bh/2))
                x2=min(w,x1+bw); y2=min(h,y1+bh)
                if x2>x1 and y2>y1: boxes.append((x1,y1,x2,y2))
                if len(boxes)>=max_regions: return boxes
    return boxes

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--threshold',type=float,default=.85)
    p.add_argument('--regions',type=int,default=25)
    args=p.parse_args()
    if not MODEL_PATH.exists(): raise FileNotFoundError(str(MODEL_PATH))
    model=tf.keras.models.load_model(MODEL_PATH)
    cap=cv2.VideoCapture(args.camera)
    if not cap.isOpened(): raise RuntimeError(f'Cannot open camera {args.camera}')
    print('Press Q to quit. Sliding-window region classification demo; NOT Fast R-CNN detection.')
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            boxes=propose_regions(frame,max(1,args.regions))
            crops=np.asarray([preprocess(frame[y1:y2,x1:x2]) for x1,y1,x2,y2 in boxes],np.float32)
            scores=model.predict(crops,verbose=0)
            for (x1,y1,x2,y2),pred in zip(boxes,scores):
                cls=int(np.argmax(pred)); conf=float(np.max(pred))
                if conf<args.threshold: continue
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f'{NAMES[cls]} {conf:.2f}',(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            cv2.imshow('Fast-R-CNN educational demo (Q to quit)',frame)
            if cv2.waitKey(1)&0xff==ord('q'): break
    finally:
        cap.release(); cv2.destroyAllWindows()
if __name__=='__main__': main()
