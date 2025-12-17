from __future__ import annotations
from abc import ABC, abstractmethod
import time
from loguru import logger

class DataHandler(ABC):
    """Abstract handler : Model for all concrete handlers"""
    def __init__(self):
        self._next_handler = None

    def set_next(self, handler: DataHandler) -> DataHandler:
        self._next_handler = handler
        return handler
    
    def handle(self, context: PipelineContext) -> PipelineContext:
        handler_name = self.__class__.__name__

        start_time = time.perf_counter()

        logger.info(f"Step {handler_name} started...")

        result_context = self.process(context)

        end_time = time.perf_counter()

        duration = end_time - start_time

        context.execution_time[handler_name] = duration

        if self._next_handler:
            return self._next_handler.handle(result_context)
        return result_context
    
    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Real process implemented in children classes"""
        pass

class PipelineContext:
    """Centralize datas exchanged by concrete handlers"""
    def __init__(self):
        self.data_map = {}
        self.final_df = None
        self.metadata = {}
        self.logs = {}
        self.execution_time = {}

        logger.add("logs/pipeline_debug.log", rotation="10 MB", level="DEBUG")

class PiplelineOrchestrator:
    """Orchestrate concrete handlers"""
    def __init__(self, handlers):
        self.first_handler = handlers[0]
        current = self.first_handler

        for handler in handlers[1:]:
            current = current.set_next(handler)

    def run(self, data):
        return self.first_handler.handle()