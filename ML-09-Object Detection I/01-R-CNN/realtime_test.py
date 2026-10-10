import argparse
import cv2
from _core import load_model,detect_demo,draw

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--threshold',type=float,default=.85)
    p.add_argument('--regions',type=int,default=25,help='Smaller is faster on CPU')
    args=p.parse_args()
    model=load_model()
    cap=cv2.VideoCapture(args.camera)
    if not cap.isOpened(): raise RuntimeError(f'Cannot open camera {args.camera}')
    print('Press Q to quit. Sliding-window crop classification demo, not real-time RPN detection.')
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            preds=detect_demo(model,frame,args.threshold,args.regions)
            cv2.imshow('Region classifier demo (Q to quit)',draw(frame,preds))
            if cv2.waitKey(1)&0xff==ord('q'): break
    finally:
        cap.release()
        cv2.destroyAllWindows()
if __name__=='__main__': main()
