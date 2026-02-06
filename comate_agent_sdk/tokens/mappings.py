# Mapping from model_name to LiteLLM model name
# Only needed for models with non-standard naming conventions

MODEL_TO_LITELLM: dict[str, str] = {
	'gemini-flash-latest': 'gemini/gemini-flash-latest',
}
