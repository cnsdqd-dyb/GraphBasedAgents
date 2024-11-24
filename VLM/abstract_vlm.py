from abc import ABC, abstractmethod


class AbstractVLM(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod
    def evaluate_states(self, states):
        pass

    @abstractmethod
    def generate(self, prompt_before_image: [str] or str, image_path: str, prompt_after_image: [str] or str, system_prompt: str, max_tokens: int,
                                   temperature: float, k: int, stop, cache_enabled: bool, api_model: str,
                                   check_tags: list, json_check: bool, stream: bool):
        pass
    