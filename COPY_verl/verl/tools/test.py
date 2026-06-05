import cv2
from modelscope.pipelines import pipeline

model_dir = '/data/zhuhairui/GroundingDINO'

pipe = pipeline('grounding-dino-task', model=model_dir)
inputs = {
    "IMAGE_PATH":"/data/zhuhairui/LongCat-Image-Edit/assets/test.png",
    "TEXT_PROMPT":"cat . dog . animal . person .",
    "BOX_TRESHOLD":0.35,
    "TEXT_TRESHOLD":0.25
}
output = pipe(inputs)
print(output['boxes'])
# from rapidocr_onnxruntime import RapidOCR

# engine = RapidOCR()

# img_path = '/data/zhuhairui/verl/verl/tools/output_0.png'
# result, elapse = engine(img_path, use_det=False, use_cls=False, use_rec=True)
# print(result)