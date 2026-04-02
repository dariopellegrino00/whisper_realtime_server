# Whisper\_Realtime\_Server


> [!WARNING]
> Most of the information in this README is still work in progress; `whisper_realtime_server` is under development.

## Installation

You can either build the server using Docker or set up a custom environment, but the NVIDIA Developer Kit is required to run the server in any configuration.

### NVIDIA Developer Kit

The NVIDIA Developer Kit is required for GPU support.
The server has been tested with CUDA 12.x and cuDNN 9, as specified in the Dockerfile.
The Whisper Streaming project has been tested with CUDA 11.7 and cuDNN 8.5.0, so it is
recommended to use at least CUDA 11.7 and cuDNN 8.5.0.

<details>
   <summary><h2>Building with Docker</h2></summary>

#### Prerequisites

Make sure Docker is installed. Follow the official [Docker Installation Guide](https://docs.docker.com/get-docker/) if needed.

Clone the repository:

```bash
git clone https://github.com/dariopellegrino00/whisper_realtime_server.git
```

#### Before building with Docker

1. Navigate to the project root directory:

   ```bash
   cd whisper_realtime_server
   ```

- When built, you will be able to test the server with your microphone or with a simulation of
  real-time audio streaming using audio files.
- There are already two audio examples in the `resources` folder. If you want to add new ones,
  do so **before** the next steps by placing the audio files in
  `whisper_realtime_server/resources`.

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
   - You can change the port mapping `50051:50051` if needed; this is the default server port.
     You can also customize the Docker startup command parameters. Check the available arguments
     in the **Running the server** section below. Fallback logic is enabled by default; pass
     `--no-fallback` if you want to disable it.
   - The server is now running and ready to accept connections. You can access it on port
     `50051` using the test client in `tools/grpc_client.py`.

4. To stop the Docker container:

   ```bash
   docker stop whisper_server
   ```

5. To restart the Docker container:

   ```bash
   docker start whisper_server
   ```
</details>
<details>
   <summary><h2>Custom Environment</h2></summary>

Install all dependencies:
- I suggest using Python environments: [Python Environments](https://docs.python.org/3/library/venv.html)
- Check the Dockerfile for additional OS packages you may need, mainly for client-side microphone support. (Linux only)

1. Install the requirements.txt:
   In the project root directory execute
   ```bash
   pip install -r requirements.txt
   ```

   If you update dependencies in `pyproject.toml`, regenerate `uv.lock` and `requirements.txt` with:
   ```bash
   python scripts/deps.py sync
   ```

   The codebase uses Ruff for formatting and linting, and mypy for type checking:
   ```bash
   uv run ruff format swim tools scripts tests
   uv run ruff check swim tools scripts tests
   uv run mypy
   ```

2. Generate the Python gRPC files:
   ```bash
   python scripts/proto.py generate
   ```
</details>

## Repository Layout

- `swim/asr/`: ASR backends and model-facing code.
- `swim/runtime/`: shared realtime runtime, processors, batching, and buffering.
- `swim/transports/grpc/`: gRPC server, sessions, stream utilities, and generated protobuf modules.
- `tools/`: manual and test-oriented utilities such as the gRPC client.
- `scripts/`: development scripts, including `proto.py` for protobuf generation and `deps.py` for dependency sync/check.

## gRPC client

<details>
     <summary><h3>Running the test client using docker</h3></summary>

If you followed the `Building with Docker` section and you want to run the client directly in the Docker container, follow these steps:

1. Ensure the container is running:

   ```bash
   docker ps 
   ```

   If you see `whisper_server` listed, you are good to go; otherwise start the container:

   ```bash
   docker start whisper_server
   ```

2. Open a terminal in the container
   
   ```bash
   docker exec -it whisper_server /bin/bash 
   ```

   Now you should see something like:
   
   ```bash
   root@<imageid>:/app# 
   ```

   Now you can run the client following the next step.
   </details>

   ### Running the client

   If you set up a custom environment, navigate to the project root directory (see the previous step if you want to run the client directly in the Docker container).

   The test client supports live microphone input on Linux and Windows.
   Run the client using your system microphone:

      ```bash
      python -m tools.grpc_client
      ```

   All the possible options:
   ```
   --host HOST          gRPC server address
   --port PORT          gRPC server port
   --all-updates        Print each response packet in a grouped block, including interim updates
   --simulate SIMULATE  Simulation mode: Path to the audio file to simulate a realtime audio
                        stream with
   --live-preview       Display confirmed and interim text in a live preview view
   --chunk-duration     Change the chunk duration (in seconds) for the audio stream
   --sound-device-id    Select a specific input device for live audio capture
   ```

   The first streaming message must be a session config. The client sends `chunk_duration` there, and the server accepts values up to `--max-chunk-duration-seconds` (default `1.0`). In practice, a `chunk_duration` around `0.4s` to `1.0s` works best for this shared realtime setup.

      Example with confirmed-only output:

      Standard output:
      ```
      0000 0600 Hi my names
      1000 2300 is Dario, nice
      3000 4500 to meet you.
      5000 7000 How are you?
      ``` 

      `--all-updates` output:
      ```
      --------------------
      CONF 1000 2300 is Dario,
      INT  2300 2600 nice
      
      --------------------
      INT  2300 3200 nice to meet
      ```

      `--live-preview` keeps the confirmed transcript on screen and renders the current interim
      continuation in light orange. To get more frequent responses with a low client count
      (roughly 1 to 5), set `chunk_duration` to `0.5`.

## gRPC Server

   ### Running the server
   ```bash
   python -m swim.transports.grpc <options>
   ```
   The server is running and ready to accept connections. You can later customize the model,
   behavior, and other options using command-line arguments. Check `--help` for more details:
   ```
   --no-fallback         Disable fallback logic when similarity local agreement
                        fails multiple times
   --fallback-threshold FALLBACK_THRESHOLD
                        threshold t for fallback logic after t+1 similarity local
                        agreement fails (ignored if fallback is disabled)
   --qratio-threshold QRATIO_THRESHOLD
                        Threshold for qratio to confirm and insert new words
                        using the hypothesis buffer (between 0 and 100), lower
                        values than 90 are not recommended
   --dedup-threshold DEDUP_THRESHOLD
                        Threshold for qratio to deduplicate overlapping words
                        between committed and new words in the hypothesis
                        buffer (between 0 and 100)
   --buffer-trimming-sec BUFFER_TRIMMING_SEC
                        Buffer trimming is the threshold in seconds that triggers
                        the service processor audio buffer to be trimmed. This is
                        useful to avoid memory leaks and to keep the buffer size
                        under control. Default value is 15 seconds
   --max-chunk-duration-seconds MAX_CHUNK_DURATION_SECONDS
                        Maximum chunk duration accepted from the client session
                        config (default: 1.0)
   --ports PORTS [PORTS ...]
                        Ports to run the server on
   --max-workers MAX_WORKERS
                        Max workers for the server
   --log-every-processor
                        Write one log file per stream. Debug-only: busy or
                        long-running servers can keep many log files open and
                        create many files.
   --model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo,turbo}
                        Name of the Whisper model to use (default:
                        large-v3-turbo). The model is automatically downloaded from the
                        model hub if not present in model cache dir
   --model-cache-dir MODEL_CACHE_DIR
                        Directory for the whisper model caching
   --model-dir MODEL_DIR
                        Directory for a custom ct2 whisper model skipping if
                        --model provided
   --warmup-file WARMUP_FILE
                        File to warm up the model and speed up the first request
   --lan LAN             Language for the whisper model to translate to (unused at
                        the moment)
   --no-vad              Disable the shared VAD preprocessing step before
                        transcription
   --log-level LOG_LEVEL
                        Log level for the server and shared ASR logger (DEBUG,
                        INFO, WARNING, ERROR, CRITICAL)
   ```
   Use `--log-every-processor` only for focused debugging. It creates one file
   handler per active stream, so busy or long-running servers can keep many log
   files open and generate a large number of files over time.

   On Windows, stop the server with `Ctrl+C`. The local server still requires a working CUDA 12 / cuDNN 9 setup, just like the Docker image.

   The server supports both `plain` and `batched` shared ASR backends through `--backend`. Shared VAD preprocessing is enabled by default and can be disabled with `--no-vad`. During testing, some issues emerged with the `batched` backend in specific scenarios, especially in repetition-heavy or timestamp-sensitive flows. For this reason, `plain` is the current default backend, while `batched` remains available for performance evaluation and further investigation.

## Documentation

> [!IMPORTANT]
> TODO: Add more documentation

Before setting up your own client, it's important to understand the server architecture. The
client first connects to a gRPC server on the default port (`50051`). After connecting, the
gRPC server assigns a service to the client. The client then streams audio data to this port
and receives real-time transcriptions.

At the moment the server runs a single shared `ParallelRealtimeASR` backend for all active
streams. Each client gets its own stream processor state, but ready processors are grouped
into the same shared inference cycle. A single gRPC streaming service is exposed, returning
confirmed transcript segments together with optional interim updates. With the current
protocol, the first streaming message is a config carrying the client `chunk_duration`. In
practice, clients connected to the same server instance should use the same
`chunk_duration`, and that duration should stay at or below the server
`--max-chunk-duration-seconds` limit.

The shared realtime backend expects connected clients to keep the declared chunk cadence. During silence, clients should keep sending silent chunks at the same cadence instead of pausing the stream. If a client stops producing chunks while remaining connected, it may be excluded so that it does not delay the other active streams.


## Credits

- This project uses parts of the Whisper Streaming project. Other projects involved in whisper streaming are credited in their repo, check it out: [whisper streaming](https://github.com/ufal/whisper_streaming)
- Credits also to: [faster whisper](https://github.com/SYSTRAN/faster-whisper)
