from kaldifeat.feature import compute_mfcc_feats
import onnxruntime as rt
import numpy as np
import pyaudio
# from jupyterplot import ProgressPlot
import threading

class Stream_VAD_Infer(object):
    def __init__(self,
                 model_path,
                 sample_rate,
                 frame_length_ms,
                 frame_shift_ms,
                 silence_frame_duration):
        self.sample_rate = sample_rate
        self.frames_length_ms = frame_length_ms
        self.frames_shift_ms = frame_shift_ms
        self.silence_frame_duration = silence_frame_duration

        # offset_ms = 20ms
        offset_ms = self.get_offset(frame_length_ms, frame_shift_ms)
        self.offset = int(offset_ms / 1000 * sample_rate)
        self.wav = np.zeros([self.offset, ])

        # Load VAD
        EP_list = ['CPUExecutionProvider']
        self.vad_ort_session = rt.InferenceSession(model_path, providers=EP_list)
        self.cnn_cache = np.zeros((1, 3, 96, 14), dtype=np.float32)

    # Data process
    def get_seg_wav(self, wav):
        self.wav = np.concatenate([self.wav, wav], axis=0)
        seg = self.wav[:]
        self.wav = self.wav[-self.offset:]
        return seg

    def get_offset(self, frame_length_ms, frame_shift_ms):
        offset_ms = 0
        while offset_ms + frame_shift_ms < frame_length_ms:
            offset_ms += frame_shift_ms
        return offset_ms

    # Extract feature
    def mfcc(self, wave):
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
                sample_frequency=self.sample_rate,
                snip_edges=True,
                use_energy=False,
                window_type='povey',
                dtype=np.float32)
        return feature

    # Infer vad
    def vad_infer(self, wave):
        wave = self.get_seg_wav(wave) * (2**15)
        feature = self.mfcc(wave)
        feature = np.array(feature, dtype=np.float32)

        ort_inputs = {self.vad_ort_session.get_inputs()[0].name: np.expand_dims(feature, 0),
                      self.vad_ort_session.get_inputs()[1].name: self.cnn_cache}
        ort_outs = self.vad_ort_session.run(None, ort_inputs)
        vad_out = np.array(ort_outs[0])
        self.cnn_cache = np.array(ort_outs[1])
        return vad_out


class Pyaudio_Stream():
    def __init__(self, stream_vad_model, chunk_size, sample_rate):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_size)
        self.continue_recording = True
        self.stream_vad_model = stream_vad_model
        self.audio = pyaudio.PyAudio()

    # Provided by Alexander Veysov
    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def stop(self):
        input("Press Enter to stop the recording:")
        self.continue_recording = False

    def start_recording(self):
        stream = self.audio.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=self.sample_rate,
                            input=True,
                            frames_per_buffer=self.chunk_size)

        # pp = ProgressPlot(plot_names=["Light VAD"],
        #                   line_names=["speech probabilities"],
        #                   x_label="audio chunks")

        stop_listener = threading.Thread(target=self.stop)
        stop_listener.start()

        while self.continue_recording:
            audio_chunk = stream.read(self.chunk_size)  # byte type
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = self.int2float(audio_int16)
            # get the confidences and add them to the list to plot them later
            new_confidence = self.stream_vad_model.vad_infer(audio_float32)
            new_confidence = np.squeeze(new_confidence, 0)[:, -1].reshape(-1)
            print(new_confidence)
            # pp.update(new_confidence)
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        # pp.finalize()


if __name__ == "__main__":
    import sys
    stream_vad_infer = Stream_VAD_Infer(model_path=sys.argv[1], sample_rate=16000,
                                        frame_length_ms=25, frame_shift_ms=10,
                                        silence_frame_duration=0)
    pyaudio_stream = Pyaudio_Stream(stream_vad_infer, chunk_size=0.2, sample_rate=16000)
    pyaudio_stream.start_recording()

