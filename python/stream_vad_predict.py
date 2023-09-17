from kaldifeat.feature import compute_mfcc_feats
import os
import argparse
import soundfile
from tqdm import tqdm
import textgrid
import onnxruntime as rt
import numpy as np


def mfcc(path, window_unit):
    if os.path.splitext(path)[-1] == ".wav":
        wave, sample_rate = soundfile.read(path) 
        wave = wave * (2**15)
    else:
        raise ValueError("Only .wav type!!!")

    feature = compute_mfcc_feats(                        
            waveform=wave,
            blackman_coeff=0.42,
            cepstral_lifter=22,
            dither=0.0,
            energy_floor=0.0,
            frame_length=25,
            frame_shift=10,
            high_freq=-200,
            low_freq=40,
            num_ceps=40,
            num_mel_bins=40,
            preemphasis_coefficient=0.97,
            raw_energy=True,
            remove_dc_offset=True,
            round_to_power_of_two=True,
            sample_frequency=sample_rate,
            snip_edges=True,
            use_energy=False,
            window_type='povey',
            dtype=np.float32)
    start = 0
    while start < len(feature):
        end = min(len(feature), start + window_unit)
        yield np.array(feature[start:end], dtype=np.float32)
        start += window_unit


def convert_frames_to_segments(frames,
                               hop_ms=10,
                               window_ms=25):
    segments = []
    is_voice = False
    switch_to_speech = False
    switch_to_non_speech = False
    frame_index = None
    start_time = None
    end_time = None
    for frame_index, value in enumerate(frames):
        if value == 1 and not is_voice:
            switch_to_speech = True
            is_voice = True
        if value == 0 and is_voice:
            switch_to_non_speech = True
            is_voice = False

        if switch_to_speech:
            start_time = frame_index * hop_ms
            switch_to_speech = False
        if switch_to_non_speech:
            end_time = (frame_index - 1) * hop_ms + window_ms
            switch_to_non_speech = False

        if start_time is not None and end_time is not None:
            segments.append((float(start_time/1000), float(end_time/1000)))
            start_time = None
            end_time = None
    if frame_index is not None and is_voice:
        end_time = frame_index * hop_ms + window_ms
        segments.append((float(start_time/1000), float(end_time/1000)))
    return segments


def convert_segments_to_textgrid(segments, textgrid_path):
    try:
        tg = textgrid.TextGrid(minTime=0, maxTime=segments[-1][-1])
        tier_word = textgrid.IntervalTier(name="non/speech",
                                          minTime=0., maxTime=segments[-1][-1])
        for i in segments:
            interval = textgrid.Interval(minTime=i[0], maxTime=i[1], mark="speech")
            tier_word.addInterval(interval)
        tg.tiers.append(tier_word)
        name_textgrid = textgrid_path + ".textgrid"
        if os.path.exists(name_textgrid):
            os.remove(name_textgrid)
        tg.write(name_textgrid)

    except:
        print("Segments is None !!!")


def get_args():
    parser = argparse.ArgumentParser(description='recognize with streaming vad model')
    parser.add_argument('--wav_path', required=True, help='wav path')
    parser.add_argument('--model_path', required=True, help='vad model file')
    parser.add_argument('--window_unit',
                        type=int,
                        default=30,
                        help='input one window length, ')
    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help='threshold')
    parser.add_argument('--sample_rate',
                        type=int,
                        default=16000,
                        help='sample_rate')
    parser.add_argument('--save_result_dir',
                        type=str,
                        default="test",
                        help='save result dir')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    # Load vad model
    EP_list = ['CPUExecutionProvider']
    vad_ort_session = rt.InferenceSession(args.model_path, providers=EP_list)

    # Infer start
    outs = []
    # batch, layer, time, channel
    cnn_cache = np.zeros((1, 3, 96, 14), dtype=np.float32)
    for feature in mfcc(args.wav_path, args.window_unit):
        ort_inputs = {vad_ort_session.get_inputs()[0].name: np.expand_dims(feature, 0),
                      vad_ort_session.get_inputs()[1].name: cnn_cache}
        ort_outs = vad_ort_session.run(None, ort_inputs)
        outs.append(np.array(ort_outs[0]))
        cnn_cache = np.array(ort_outs[1])
    outs = np.concatenate(outs, 1)  # batch, time, 2
    pro_frame_probabilities = np.squeeze(outs, 0)[:, -1].reshape(-1)
    single_frame_predictions = [1 if i > args.threshold else 0 for i in pro_frame_probabilities]

    
    if not os.path.exists(args.save_result_dir):
        os.mkdir(args.save_result_dir)
    wav_scp = open(args.save_result_dir + "/wav.scp", "w")
    wav_name = os.path.splitext((os.path.basename(args.wav_path)))[0]
    save_path = os.path.join(args.save_result_dir, wav_name)

    segments = convert_frames_to_segments(frames=single_frame_predictions)

    # save predict probability
    prob_time = [[indexs * 10, i] for indexs, i in enumerate(pro_frame_probabilities)]  # hop_ms=10
    if os.path.exists(save_path + ".prob"):
    	os.remove(save_path + ".prob")
    np.savetxt(save_path + ".prob", np.array(prob_time), fmt='%.4f')

    # save segments
    for i in segments:
        wav_scp.write(wav_name + ' ' + wav_name + '_' + str(int(i[0] * 1000)) + '_' + str(
            int(i[1] * 1000)) + ' ' + str(i[0]) + ' ' + str(i[1]) + "\n")
    
    # save textgrid
    convert_segments_to_textgrid(segments, save_path)
    print("Successfully completed !!!")
if __name__ == '__main__':
    main()


