import logging
from src.server.stream_utils import *
from src.generated import speech_pb2 as whisp_speech
from abc import ABC, abstractmethod
from typing import Dict, List

class StreamSession(ABC): 
    """
    Abstract Class to manage the stream session. 
    Manage the state of each gRPC service. 
    """

    def __init__(self, processor_manager: ProcessorManager, server_logger=None, logger=None):
        self.processor_manager = processor_manager
        self.id = processor_manager.id
        self.server_logger = server_logger if server_logger is not None else logging.getLogger(__name__) 
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    @abstractmethod #
    async def request_enqueuer(self, request_iterator):
        """
        Enqueue the requests from the request iterator using an asyn audio queue
        """
        pass

    @abstractmethod
    async def manage_first_message(self, first_request):
        """
        Manage the first message of the stream session. sometimes its the config, sometimes the first audio chunk.
        """
        pass

    @abstractmethod
    def create_response(self) -> List: # compatible format for protofiles response 
        """
        Format the response for the stream session based on the transcription managers rules.
        """
        pass

    @abstractmethod
    def final_response(self) -> List: # grpc compatible reponse to protofiles
        """
        Custom behavior for the final response for the stream session, often the same as format_response.
        """
        pass

# the services that derives from this class are the ones that 
# adaps whisper streaming return format and some other logic
class WhispStreamSession(StreamSession):
    """
    Abstract Whisper Stream Session class to manage the stream session for the whisper services.
    """

    def __init__(self, processor_manager: ProcessorManager, server_logger=None, logger=None):
        super().__init__(processor_manager, server_logger, logger)
        self.transcription_managers = self.create_transcription_managers()

    def create_transcription_managers(self) -> Dict[str, TranscriptionManager]:
        """
        Create the transcription managers for the stream session. 
        This is a placeholder method to be implemented by the subclasses.
        """
        return {"confirmed": TranscriptionManager()}

    async def request_enqueuer(self, request_iterator):
        try:
            async for audio_chunk in request_iterator:
                audio_samples = np.frombuffer(audio_chunk.audio_bytes, dtype=np.float32)
                await self.processor_manager.audio_queue.put(audio_samples)
        except Exception as e:
            self.processor_manager.logger.error(f"Exception in request_enqueuer {self.processor_manager.id}: {e}")

    # this method is called to handle the first message of the stream session.
    # purpose is allow the session to handle eventual configuration messages in the future. with easier extensibility for more session types.
    async def manage_first_message(self, first_request): 
        audio_samples = np.frombuffer(first_request.audio_bytes, dtype=np.float32)
        await self.processor_manager.insert_audio(audio_samples) 

class StandardWhispStreamSession(WhispStreamSession):
    """
    Implementation of the original Whisper Streaming server response format using gRPC and the Shared ASR architecture

    """

    def create_response(self):
        results = self.processor_manager.processor.results
        exist, fmt = self.transcription_managers["confirmed"].format_transcript(results)
        return [self._create_response(*fmt)] if exist else []

    def final_response(self):
        results = self.processor_manager.processor.finish()
        exist, fmt = self.transcription_managers["confirmed"].format_transcript(results)
        return [self._create_response(*fmt)] if exist else []

    def _create_response(self, start, end, text):
        return whisp_speech.Transcript(
            start_time_millis=start,
            end_time_millis=end,
            text=text
        )

class HypothesisWhispStreamSession(WhispStreamSession):
    """
    Variant of the whisp stream session that includes the hypothesis buffer
    """

    def create_transcription_managers(self):
        return {"confirmed": TranscriptionManager(), "hypothesis": TranscriptionManager()}

    def create_response(self):
        pass
        results = self.processor_manager.processor.results
        hypothesis = self.processor_manager.processor.hypothesis
        exist1, fmt_t = self.transcription_managers["confirmed"].format_transcript(results)
        exist2, fmt_h = self.transcription_managers["hypothesis"].format_transcript(hypothesis)
        return [self._create_response(*fmt_t, *fmt_h)] if exist1 or exist2 else []

    def final_response(self):
        results = self.processor_manager.processor.finish()
        exist, fmt_t = self.transcription_managers["confirmed"].format_transcript(results)
        return [self._create_response(*fmt_t, 0, 0, "")] if exist else []


    def _create_response(self, start_t, end_t, text, start_h, end_h, hypothesis):
        return whisp_speech.TranscriptWithHypothesis(
            confirmed= whisp_speech.Transcript(
                start_time_millis=start_t,
                end_time_millis=end_t,
                text=text
            ),
            hypothesis=whisp_speech.Transcript(
                start_time_millis=start_h,
                end_time_millis=end_h,
                text=hypothesis
            )
        )

