from absl import app
import cv2
from starlette.responses import StreamingResponse
import io
import shutil
import os
import re
from inference import inference
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import time
from df2NVHExpert import df2NVHExpert
from azure_client import fsAzureStorage

path2label_map_json = r'./label_map.json' 

model_path = r'./FaurSound_model/saved_model_image_input_v1-4'
model_version = re.search(r'v\d.\d.?.?',model_path).group(0)

inferenceAPI = inference(path2label_map_json = path2label_map_json, model_path = model_path)

azureClient = fsAzureStorage(model_version = model_version)

OUTPUT_PATH = r'./detections'   # path to output folder where images with detections are saved

app = FastAPI()


@app.post("/EOL/")
async def inference_with_EOL_output(wav: UploadFile = File(...)):

    t1 = time.time()
    wav_name = wav.filename
    title, _ = os.path.splitext(wav_name)
    temp_folder = r'./temp'
    wav_file_path = os.path.join(temp_folder, wav_name)
    try:
        with open(wav_file_path,'wb+') as buffer:
            shutil.copyfileobj(wav.file, buffer)
    finally:
        wav.file.close()

    t2 = time.time()
    print('save wav file time: {}'.format(t2 - t1))

    inferenceAPI.get_spec_pics_for_wav(wav_file_path,output_folder=temp_folder , get_infor_from_fn=True)
    cv_file_path = os.path.join(temp_folder,'cv',f'{title}.jpg')
    stft_file_path = os.path.join(temp_folder,'stft',f'{title}.jpg')

    t3 = time.time()
    print('save stft/cv file time: {}'.format(t3 - t2))

    img = inferenceAPI.inference_one_img_with_plot_on_stft(cv_file_path,stft_file_path,
                                                save_png_folder=OUTPUT_PATH,save_txt=True)
    
    predict_pic_fp = os.path.join(OUTPUT_PATH, f'{title}.png')

    txt_dir_path=os.path.join(os.path.dirname(cv_file_path),'prediction_txt_raw_output')
    txt_file_path = os.path.join(txt_dir_path, f'{title}.txt')

    t4 = time.time()
    print('inferenceAPI time: {}'.format(t4 - t3))

    # prepare image for response
    im_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode('.png', im_BGR)
    # response = img_encoded.tostring()

    azureClient.upload_to_azure(cv_file_path,txt_file_path,predict_pic_fp)
    #remove temporary files
    os.remove(wav_file_path)
    os.remove(cv_file_path)
    os.remove(stft_file_path)
    os.remove(predict_pic_fp)
    os.remove(txt_file_path)
    t5 = time.time()
    print('prepare output and upload to Azure time: {}'.format(t5 - t4))
    print('TOTAL time: {}'.format(t5 - t1))

    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")



# @app.route('/EOL/cl', methods=['POST'])
# def inference_with_EOL_raw_output():
#     wav_file = request.files["wav"]
#     wav_name = wav_file.filename
#     title, _ = os.path.splitext(wav_name)
#     cwd_path = os.getcwd()
#     wav_file_path = os.path.join(cwd_path, wav_name)
#     wav_file.save(wav_file_path)
#     wav_file.close()

#     inferenceAPI.get_spec_pics_for_wav(wav_file_path,output_folder=cwd_path , get_infor_from_fn=True)
#     cv_file_path = os.path.join(cwd_path,'cv',f'{title}.jpg')
#     stft_file_path = os.path.join(cwd_path,'stft',f'{title}.jpg')

#     _ = inferenceAPI.inference_as_raw_output(cv_file_path, to_file = True)

#     txt_output_folder = r'./cv/prediction_txt_raw_output'
#     stft_folder = os.path.dirname(stft_file_path)
#     NVH = df2NVHExpert(path2prediction = txt_output_folder,path2stft = stft_folder,
#             path2label_map_json = path2label_map_json, output_path = r'./detections/cleaned_csv')

#     output = NVH.return_json()
#     parsed = json.loads(output)
#     json_str = json.dumps(parsed, indent=4)

#     #remove temporary image
#     os.remove(wav_file_path)
#     os.remove(cv_file_path)
#     os.remove(stft_file_path)
#     for file in os.listdir(txt_output_folder):
#         os.remove(os.path.join(txt_output_folder,file))

#     try:
#         # return jsonify({"response":json_str}), 200
#         return json_str, 200
#     except FileNotFoundError:
#         abort(404)


@app.get('/hello/')
def hello():
    faursound_api_github = r'https://github.com/WangCHEN9/faursound_core'
    response = f'Hello, see how to use this API in {faursound_api_github} !'
    return response
