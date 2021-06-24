from absl import app
import cv2
from starlette.responses import StreamingResponse
import io
# import shutil
import os
import re
import json
from inferenceEOL import inferenceEOL
from fastapi import FastAPI, File, UploadFile
import time
from df2NVHExpert import df2NVHExpert
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


path2label_map_json = r'./label_map.json' 

model_path = r'./FaurSound_model/saved_model_image_input_v1-4'
model_version = re.search(r'v\d.\d.?.?',model_path).group(0)

inferenceAPI = inferenceEOL(path2label_map_json = path2label_map_json, model_path = model_path)

OUTPUT_PATH = r'./detections'   # path to output folder where images with detections are saved

app = FastAPI(
        title="FaurSound API",
        version=model_version,
        description=f"noise analysis AI for etilt EOL",
)
FastAPIInstrumentor.instrument_app(app)

def update_log_file(log_file, infor:str):
    with open(log_file, "a") as f:
        f.write(infor) 
        f.write("\n")

@app.post("/EOL/")
async def inference_with_EOL_output(wav: UploadFile = File(...)):

    t1 = time.time()
    wav_name = wav.filename #! we use input from EOL to create filename when link with labview

    title, _ = os.path.splitext(wav_name)
    temp_folder = r'./temp'
    t2 = time.time()
    print('save wav file time: {}'.format(t2 - t1))

    inferenceAPI.get_spec_pics_for_wav(wav.file,output_folder=temp_folder ,fn=wav_name)
    cv_file_path = os.path.join(temp_folder,'cv',f'{title}.jpg')
    stft_file_path = os.path.join(temp_folder,'stft',f'{title}.jpg')

    t3 = time.time()
    print('save stft/cv file time: {}'.format(t3 - t2))

    img = inferenceAPI.inference_one_img_with_plot_on_stft(cv_file_path,stft_file_path,
                                                save_png_folder=OUTPUT_PATH,save_txt=True)
    
    # predict_pic_fp = os.path.join(OUTPUT_PATH, f'{title}.png') 

    txt_dir_path=os.path.join(os.path.dirname(cv_file_path),'prediction_txt_raw_output')
    txt_file_path = os.path.join(txt_dir_path, f'{title}.txt')

    t4 = time.time()
    print('inferenceAPI time: {}'.format(t4 - t3))

    im_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode('.png', im_BGR)

    stft_folder = os.path.dirname(stft_file_path)
    NVH = df2NVHExpert(path2prediction = txt_dir_path,path2stft = stft_folder,
                        path2label_map_json = path2label_map_json)

    output = NVH.return_json_from_one_txt_full(txt_file = f'{title}.txt')
    parsed = json.loads(output)
    json_str = json.dumps(parsed, indent=4)

    wav.file.close()
    os.remove(cv_file_path)#? cv file to be saved
    os.remove(stft_file_path)
    # os.remove(predict_pic_fp)#? prediction file to saved
    os.remove(txt_file_path) #? txt file to be saved
    t5 = time.time()
    print('prepare output time: {}'.format(t5 - t4))
    print('TOTAL time: {}'.format(t5 - t1))
    #! log response time
    update_log_file(f'./log/server_log.txt', str(round((t5 - t1),3)))
    # return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")
    return json_str


@app.get('/hello/')
def hello():
    faursound_api_github = r'https://github.com/WangCHEN9/faursound_core'
    response = f'Hello, see how to use this API in {faursound_api_github} !'
    return response

