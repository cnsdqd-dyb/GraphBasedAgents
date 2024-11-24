from abc import ABC, abstractmethod


class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod
    def evaluate_states(self, states):
        pass

    @abstractmethod
    def generate(self, system_prompt: str, example_prompt: [str] or str, max_tokens: int,
                                   temperature: float, k: int, stop, cache_enabled: bool, api_model: str,
                                   check_tags: list, json_check: bool, stream: bool):
        pass
    
    # @abstractmethod
    # def batch_generate(self, system_prompt: str, user_prompts: [str] or str, example_prompts: [str] or str, max_tokens: int, temperature: float,
    #                     k: int, stop, cache_enabled: bool, api_model: str, check_tags: list, json_check: bool):
    #     '''
    #     system_prompt: the first prompt to describe the model
    #     user_prompts: the prompts from the user
    #     example_prompts: the prompts from the example which can be used to guide the model
    #     max_tokens: the maximum number of tokens to generate
    #     '''
    #     pass