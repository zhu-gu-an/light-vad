python stream_vad_predict.py  --wav_path test.wav \
       			      --model_path model/16k_stream_vad.onnx  \
			      --window_unit 30 \
			      --threshold 0.5 \
			      --sample_rate 16000 \
			      --save_result_dir test 
