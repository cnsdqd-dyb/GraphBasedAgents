
def init_vlm(args: dict):
    api_model = args.get("api_model", "")

    if "gpt" in api_model:
        from VLM.openai_models import OpenAILanguageModel
        new_args = {
            "api_key": args.get("api_key", None),
            "api_model": api_model,
            "api_base": args.get("api_base", None),
            "evaluation_strategy": args.get("evaluation_strategy", None),
            "enable_ReAct_prompting": args.get("enable_ReAct_prompting", None),
            "strategy": args.get("strategy", None),
            "role_name": args.get("role_name", None),
            "api_key_list": args.get("api_key_list", None)
        }
        new_args = {k: v for k, v in new_args.items() if v is not None}
        return OpenAILanguageModel(**new_args)
    