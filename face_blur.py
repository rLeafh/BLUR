import cv2, os, sys, torch
from ultralytics import YOLO

MODEL_PATH   = "yolov8n-face.pt"
CONF_THRESH  = 0.40
IMG_SIZE     = 1280
OUTPUT_NAME  = "saida_censurada.mp4"

def clamp(v, a, b): return max(a, min(b, v))

def blur_faces(input_path: str, output_name: str = OUTPUT_NAME):

    if not os.path.isfile(input_path):
        sys.exit(f"Arquivo não encontrado: {input_path}")

    USE_GPU  = torch.cuda.is_available()
    DEVICE   = 0 if USE_GPU else "cpu"
    USE_FP16 = USE_GPU

    print(f"Dispositivo selecionado: {'GPU' if USE_GPU else 'CPU'}")

    model = YOLO(MODEL_PATH).to(DEVICE)
    model.fuse()                    

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit("Erro ao abrir o arquivo de vídeo.")

    fps, w, h = (cap.get(cv2.CAP_PROP_FPS),
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_name, fourcc, fps, (w, h))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        res = model.predict(frame,
                            device=DEVICE,
                            imgsz=IMG_SIZE,
                            conf=CONF_THRESH,
                            iou=0.5,
                            half=USE_FP16,
                            verbose=False)

        for x1, y1, x2, y2 in res[0].boxes.xyxy.int().tolist():
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1, x2 = clamp(x1,0,w-1), clamp(x2,0,w-1)
            y1, y2 = clamp(y1,0,h-1), clamp(y2,0,h-1)
            roi = frame[y1:y2, x1:x2]
            k = int(min(x2 - x1, y2 - y1) / 3) | 1
            sigma = k                              
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), sigma)

        out.write(frame)
        idx += 1
        if idx % 100 == 0:
            print(f"{idx} frames...")

    cap.release(); out.release()
    print(f"Concluído! Arquivo salvo como {output_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python face_blur_yolov8_auto.py <video.mp4> [saida.mp4]")
        sys.exit(1)
    entrada = sys.argv[1]
    saida   = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_NAME
    blur_faces(entrada, saida)
