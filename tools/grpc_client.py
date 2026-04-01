#!/usr/bin/env python3
import argparse
import queue
import sys
import time

import grpc
import librosa
import numpy as np
import sounddevice as sd

from swim.transports.grpc.generated import speech_pb2, speech_pb2_grpc

# WARNING: This code is just a test client for testing the multiclient realtime whisper server


# Audio configuration
class AudioConfig:
    sample_rate = 16000  # Whisper wants 16 kHz
    chunk_duration = 1.0  # seconds
    channels = 1
    sample_format = np.float32

    @classmethod
    def chunk_duration_millis(cls):
        return int(round(cls.chunk_duration * 1000))

    @classmethod
    def effective_chunk_duration_seconds(cls):
        return cls.chunk_duration_millis() / 1000.0

    @classmethod
    def chunk_size(cls):
        return int(cls.chunk_duration_millis() * cls.sample_rate / 1000)


# gRPC client for transcription
class TranscriptorClient:
    def __init__(
        self,
        host: str,
        port: int,
        with_hypothesis: bool = False,
        simulate_filepath: str = "",
        interactive: bool = False,
    ):
        self.host = host
        self.port = port
        self.with_hypothesis = with_hypothesis
        self.simulate_filepath = simulate_filepath
        self.interactive = interactive

    def __generate_audio_chunks_sim(self):
        # Simulation mode: load audio file (using librosa)
        audio_data, sr = librosa.load(
            self.simulate_filepath,
            sr=AudioConfig.sample_rate,
            mono=True,
            dtype=AudioConfig.sample_format,
        )
        total_samples = len(audio_data)
        print(f"Loaded {total_samples} samples from file {self.simulate_filepath}")
        yield speech_pb2.StreamingRecognizeRequest(
            config=speech_pb2.StreamingConfig(
                chunk_duration_millis=AudioConfig.chunk_duration_millis()
            )
        )
        # Split into 1-second chunks
        for i in range(0, total_samples, AudioConfig.chunk_size()):
            chunk_samples = audio_data[i : i + AudioConfig.chunk_size()]
            # if len(chunk) < AudioConfig.chunk_size() and len(audio_data) - i+AudioConfig.chunk_size():
            # Skip incomplete last chunk
            # break
            yield speech_pb2.StreamingRecognizeRequest(
                audio_chunk=speech_pb2.AudioChunk(audio_bytes=chunk_samples.tobytes())
            )
            # Simulate real-time sending
            time.sleep(AudioConfig.effective_chunk_duration_seconds())

    def __generate_audio_chunks_live(self):
        print("Capturing real-time audio from the microphone...")
        device_info = sd.query_devices(sd.default.device[0])
        native_sample_rate = int(
            device_info["default_samplerate"]
        )  # Native sample rate of the device
        print(
            f"Mic sample rate: {native_sample_rate} Hz, preferred capture sample rate: {AudioConfig.sample_rate} Hz"
        )

        try:
            yield from self._generate_audio_chunks_live_stream(
                capture_sample_rate=AudioConfig.sample_rate,
                source_sample_rate=AudioConfig.sample_rate,
            )
            return
        except sd.PortAudioError as exc:
            if native_sample_rate == AudioConfig.sample_rate:
                raise
            print(f"Falling back to native microphone sample rate {native_sample_rate} Hz: {exc}")

        yield from self._generate_audio_chunks_live_stream(
            capture_sample_rate=native_sample_rate,
            source_sample_rate=native_sample_rate,
        )

    def _generate_audio_chunks_live_stream(
        self,
        capture_sample_rate: int,
        source_sample_rate: int,
    ):
        callback_blocksize = max(1, int(capture_sample_rate * 0.05))
        chunk_samples = max(
            1,
            int(AudioConfig.effective_chunk_duration_seconds() * capture_sample_rate),
        )
        audio_queue = queue.Queue(maxsize=100)

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}", file=sys.stderr)
            chunk = indata.copy().flatten()
            try:
                audio_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass

        with sd.InputStream(
            device=sd.default.device[0],
            channels=AudioConfig.channels,
            samplerate=capture_sample_rate,
            blocksize=callback_blocksize,
            dtype=AudioConfig.sample_format,
            latency="low",
            callback=audio_callback,
        ):
            yield speech_pb2.StreamingRecognizeRequest(
                config=speech_pb2.StreamingConfig(
                    chunk_duration_millis=AudioConfig.chunk_duration_millis()
                )
            )

            buffered_chunks = []
            buffered_samples = 0

            while True:
                chunk = audio_queue.get()
                buffered_chunks.append(chunk)
                buffered_samples += len(chunk)

                if buffered_samples < chunk_samples:
                    continue

                chunk_native = np.concatenate(buffered_chunks)
                if len(chunk_native) > chunk_samples:
                    chunk_to_send = chunk_native[:chunk_samples]
                    remainder = chunk_native[chunk_samples:]
                    buffered_chunks = [remainder] if len(remainder) else []
                    buffered_samples = len(remainder)
                else:
                    chunk_to_send = chunk_native
                    buffered_chunks = []
                    buffered_samples = 0

                if source_sample_rate != AudioConfig.sample_rate:
                    chunk_to_send = librosa.resample(
                        chunk_to_send,
                        orig_sr=source_sample_rate,
                        target_sr=AudioConfig.sample_rate,
                    )

                yield speech_pb2.StreamingRecognizeRequest(
                    audio_chunk=speech_pb2.AudioChunk(
                        audio_bytes=np.array(chunk_to_send, dtype=np.float32).tobytes()
                    )
                )

    def generate_audio_chunks(self):
        """
        Iteratively generates 1-second audio chunks.
        If simulate_filepath is set, the client loads the audio from a file;
        otherwise, it captures live audio from the microphone.
        Each chunk is sent as an AudioChunk message.
        """
        print("Started connection")
        if self.simulate_filepath:
            return self.__generate_audio_chunks_sim()
        else:
            return self.__generate_audio_chunks_live()

    # TODO: refactor to reduce code duplication
    def run(self):
        """
        Creates the gRPC connection, sends audio chunks via bidirectional streaming,
        and prints the transcriptions received from the server.
        In interactive mode, the transcript is updated on a single line.
        """
        if self.with_hypothesis:
            self.run_with_hypothesis()
        else:
            # Create gRPC channel and stub
            channel = grpc.insecure_channel(f"{self.host}:{self.port}")
            stub = speech_pb2_grpc.SpeechToTextStub(channel)

            # Audio chunk generator
            audio_generator = self.generate_audio_chunks()

            # Start the bidirectional call
            responses = stub.StreamingRecognize(audio_generator)
            try:
                last_resp_time = 0
                for response in responses:
                    if self.interactive:
                        # Update the transcript on the same line.
                        # Clear the line and write the updated transcript
                        if (
                            response.text
                            and response.text[-1] == "."
                            and response.start_time_millis - last_resp_time > 1000
                        ):
                            print(response.text)
                        else:
                            print(response.text, end="", flush=True)
                        last_resp_time = response.end_time_millis
                    else:
                        print(
                            f"Received transcription: {response.start_time_millis} {response.end_time_millis} {response.text}"
                        )
            except grpc.RpcError as e:
                print("gRPC Error:", e)
            finally:
                channel.close()

    def print_fullscreen(self, confirmed, hypothesis):
        # ANSI escape: clear screen + move cursor top-left
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        sys.stdout.write(confirmed)
        if hypothesis:
            sys.stdout.write(f"\033[90m{hypothesis}\033[0m")  # ipotesi in grigio
        sys.stdout.flush()

    def run_with_hypothesis(self):
        """
        Creates the gRPC connection, sends audio chunks via bidirectional streaming,
        and prints the transcriptions received from the server. This method also receive and handles the hypothesis of the transcription.
        """
        # Create gRPC channel and stub
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = speech_pb2_grpc.SpeechToTextWithHypothesisStub(channel)

        # Audio chunk generator
        audio_generator = self.generate_audio_chunks()

        # Start the bidirectional call
        responses = stub.StreamingRecognize(audio_generator)
        try:
            if self.interactive:
                confirmed_text = ""
                last_confirmed = ""
                last_hypothesis = ""

                for response in responses:
                    updated = False

                    if response.confirmed.text and response.confirmed.text != last_confirmed:
                        confirmed_text += response.confirmed.text
                        last_confirmed = response.confirmed.text
                        updated = True
                        last_hypothesis = ""

                    if response.hypothesis.text != last_hypothesis:
                        last_hypothesis = response.hypothesis.text
                        updated = True

                    if updated:
                        self.print_fullscreen(confirmed_text, last_hypothesis)
            else:
                for response in responses:
                    confirmed = response.confirmed
                    hypothesis = response.hypothesis
                    print(
                        f"Confirmed: {confirmed.start_time_millis} {confirmed.end_time_millis} \033[92m{confirmed.text}\033[00m"
                    )
                    print(
                        f"Hypothesis: {hypothesis.start_time_millis} {hypothesis.end_time_millis} \033[91m{hypothesis.text}\033[00m\n"
                    )

        except grpc.RpcError as e:
            print("gRPC Error:", e)
        finally:
            channel.close()


def main():
    parser = argparse.ArgumentParser(description="gRPC Transcriptor Client")
    parser.add_argument("--host", type=str, default="localhost", help="gRPC server address")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument(
        "--with-hypothesis",
        action="store_true",
    )
    parser.add_argument(
        "--simulate",
        type=str,
        default=None,
        help="Simulation mode: Path to the audio file to simulate a realtime audio stream with",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Display transcript updates interactively on a single line",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=1.0,
        help="Chunk duration in seconds for realtime streaming. Must be > 0",
    )
    parser.add_argument(
        "--sound-device-id",
        type=int,
        default=None,
        help="ID of the sound device to use for live audio capture. If not specified, the default device will be used. Use this option to select a specific device if you have multiple audio input devices or sounddevice is using the wrong one.",
    )
    args = parser.parse_args()
    if args.chunk_duration <= 0:
        parser.error("--chunk-duration must be > 0 seconds")
    AudioConfig.chunk_duration = args.chunk_duration
    if AudioConfig.chunk_duration_millis() <= 0:
        parser.error("--chunk-duration must round to at least 1 millisecond")
    sd.default.device = (
        args.sound_device_id if args.sound_device_id is not None else sd.default.device
    )

    client = TranscriptorClient(
        host=args.host,
        port=args.port,
        with_hypothesis=args.with_hypothesis,
        simulate_filepath=args.simulate,
        interactive=args.interactive,
    )
    client.run()


if __name__ == "__main__":
    main()
