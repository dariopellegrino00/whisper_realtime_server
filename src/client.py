from dataclasses import dataclass
from whisper_online import *
import sounddevice as sd
import numpy as np
import socket, threading, sys 

@dataclass
class AudioConfig:
    sample_rate = 16000
    chunk_duration = 1
    channels = 1
    chunk_size = int(sample_rate * chunk_duration)

@dataclass
class NetConfig:
    host = 'localhost'
    port = 12001


class TranscriptorClient:
 
    mean_time, n_responses = 0, 0
    @staticmethod 
    def __error(msg):
        sys.stderr.write(f"\033[91m" + msg + "\033[0m\n")

    @staticmethod 
    def __log(msg):
        sys.stdout.write(f"\033[96m" + msg + "\033[0m\n")

    def __send_audio(self, stream, sock):
        """Capture audio from the microphone and send it to the server."""
        try:
            while True:
                # Capture audio chunk
                audio_data = stream.read(AudioConfig.chunk_size)[0]
                audio_chunk = (audio_data[:, 0] * 32768).astype(np.int16).tobytes()
                
                # Send audio chunk to server
                sock.sendall(audio_chunk)
        except Exception as e:
            TranscriptorClient.__error(f"Error in sending audio: {e}")

    def __receive_transcriptions(self, sock):
        """Receive transcriptions from the server."""
        try:
            while True:
                response = sock.recv(1024).decode('utf-8').strip().split(' ', 2)
                self.mean_time += abs (float(response[1]) - float(response[0]))
                self.n_responses += 1

                if response:
                    print(f"{response}")
        except Exception as e:
            TranscriptorClient.__error(f"Error in receiving transcription: {e}")

    def start(self):
        """Initialize the microphone stream and handle client-server communication."""
        try:
            # Initialize microphone
            with sd.InputStream(callback=None, channels=1, samplerate=AudioConfig.sample_rate, blocksize=AudioConfig.chunk_size) as stream:
                # Connect to the server
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((NetConfig.host, NetConfig.port))
                    TranscriptorClient.__log(f"Connected to server at {NetConfig.host}:{NetConfig.port}")
                    
                    sender_thread = threading.Thread(target=self.__send_audio, args=(stream, s), daemon=True)
                    receiver_thread = threading.Thread(target=self.__receive_transcriptions, args=(s,), daemon=True)
                    sender_thread.start()
                    receiver_thread.start()
                    
                    while sender_thread.is_alive() and receiver_thread.is_alive(): 
                        threading.Event().wait(0.1)
        except KeyboardInterrupt : 
            print(f"mean time to responde {self.mean_time / self.n_responses}")
        except Exception as e:
            TranscriptorClient.__error(f"Error {e}")

if __name__ == "__main__":
    client = TranscriptorClient()
    client.start()
