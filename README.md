# Whisper\_Realtime\_Server


> [!WARNING]
> Most of the informations in README are work in progress `whisper_realtime_server` is under development.

## Installation

### Building with Docker

#### Prerequisites

Make sure Docker is installed. Follow the official [Docker Installation Guide](https://docs.docker.com/get-docker/) if needed.

Clone the repository:

```bash
git clone https://github.com/dariopellegrino00/whisper_realtime_server.git
cd whisper_realtime_server
```

#### Before builing using docker

- When built, you will be able to test the server with your mic or a simulation of realtime audio streaming using audio files
- there are already two audio examples in the resources folder, if you want to add new ones, then **BEFORE** the next steps, add the audio files to `whisper_realtime_server/resources`

#### Steps to Build and Run the Docker Image

1. Navigate to the project root directory:

   ```bash
   cd whisper_realtime_server
   ```

2. Build the Docker image:

   ```bash
   docker build -t whisper_realtime_server .
   ```

3. Run the Docker container with GPU support and port mapping:

   ```bash
   docker run --gpus all -p 50051:50051 --name whisper_server whisper_realtime_server
   ```
   - You can change the port range `50051:50051` if needed. Remember to change port on `whisper_server.py` and `gprcclient.py`.
   - The server is now running and ready to accept connections. You can access it at port `50051` using the `grpcclient.py` script.

4. To stop the Docker container:

   ```bash
   docker stop whisper_server
   ```

5. To restart the Docker container:

   ```bash
   docker start whisper_server
   ```

### Running the test client

if you want to run the client directly in the docker container follow these steps:

1. Ensure the container in running:

   ```bash
   docker ps 
   ```

   if you see whisper_server listed then you are good to go, otherwise start the container 

   ```bash
   docker start whisper_server
   ```

2. Open a terminal in the container
   
   ```bash
   docker exec -it whisper_server /bin/bash 
   ```

   now you should see something like:
   
   ```bash
   root@<imageid>:/app/src# 
   ```

3. Run the grpc client

   - Run the client using your system microphone:
      ```bash
      python3 -m src.client
      ```
   - All the possible options
      ```bash
      options:
        -h, --help           show this help message and exit
        --host HOST          gRPC server address
        --port PORT          gRPC server port
        --simulate SIMULATE  Simulation mode: Path to the audio file to simulate a realtime audio
                             stream with
        --interactive        Display transcript updates interactively on a single line
      ```

      Standard output:
      ```
      0 600 Hi my names
      1000 2300 is Dario, nice 
      3000 4500 to meet you.
      5000 7000 How are you?
      ``` 

      Interactive output:
      ```
      Hi my names is Dario, nice to meet you. 
      How are you? 
      ```

### Custom Environment

> [!IMPORTANT]
> TODO 

## Whisper Server Config File JSON Tutorial

> [!CAUTION]
> For now, avoid modifying the `config.json` file. If you need to experiment, it is advisable to only adjust the model size parameter.

## Num Workers and Token Confirmation Threshold

> [!IMPORTANT]
> TODO: tweaking tutorial and explanation

## Nvidia Developer Kit

The Nvidia Developer Kit is required for GPU support. The server has been tested with CUDA 12.X and cuDNN 9, as specified in the Dockerfile. The Whisper Streaming project has been tested with CUDA 11.7 and cuDNN 8.5.0, so it is recommended to use at least CUDA 11.7 and cuDNN 8.5.0.&#x20;

## Documentation

> [!IMPORTANT]
> TODO: Add more documentation

Before setting up your own client, it's important to understand the server architecture. The client first connects to a GRPC server on the default port (`50051`). After connecting, the GRPC server assigns a service to the client. Then the client streams audio data to this port, and receives real-time transcriptions.&#x20;

## Testing the server locally

Install all dependencies: 
- I suggest the use of python enviroments: [Python Enviroments](https://docs.python.org/3/library/venv.html)
- Check Dockerfile for addictional OS packages you may miss
- An actual better tutorial for local installations is in the TODO list

1. Install the requirements.txt:
   In the project root directory execute
   ```bash
   pip install -r requirements.txt
   ```

2. Generate the pytgon-grpc files for grpc:
   ```bash
   make proto
   ```

2. Navigate to the `src` directory:

   Inside the repository folder get in `src`, run:
   ```bash
   cd src
   ```

3. Run the server directly with Python:

   ```bash
   python3 -m src.server
   ```

4. To use a microphone for audio input:

   ```bash
   python3 -m src.client
   ```

   Check the docker section for client running options

## Credits

- This project uses parts of the Whisper Streaming project. Other projects involved in whisper streaming are credited in their repo, check it out: [whisper streaming](https://github.com/ufal/whisper_streaming)
- Credits also to: [faster whisper](https://github.com/SYSTRAN/faster-whisper)

## Contributing

This project is still in an early stage of development, and there may be significant bugs or issues. All contributions are welcome and greatly appreciated!
If you'd like to contribute, here's how you can help:

- **Fork** the repository.
- Create a **new branch** for your feature or bug fix.
- Submit a **pull request** with a clear description of your changes.

For major changes, please open an **issue** first to discuss what you'd like to change.
Thank you for helping improve this project and making it better for everyone!

## TODO
- [x] Rapidfuzz token confirmation 
- [x] grpc implementation
- [ ] Secure grpc connections
- [ ] Custom enviroment setup
- [ ] remove unused packages in Dockerfile and requirements 

## FIXED 
- [x] Server fail to always return indipendent ports on concurrent requests now fixed
- [x] Send back last confirmed token when client send silent audio for a prolonged time (or no human speech audio)
- [x] Rarely other client words can end in others buffer 
- [x] `MultiProcessingFasterWhisperASR` and the Grpc Speech to text services can get stuck with high number of streaming active concurrently (10 to 20)
- [x] ValueError(f"{id} is not a registered processor.") at end of services

## KNOWN BUGS - UNKNOWN CAUSE
- [ ] Random words like `ok` or `thank you` are transcribed when client stays silent 
